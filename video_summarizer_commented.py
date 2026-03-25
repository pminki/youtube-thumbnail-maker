"""
[영상 요약 파이프라인]
이 파일은 LangGraph를 활용해서 영상을 자동으로 요약하는 흐름을 구현합니다.
전체 흐름: 영상 → 오디오 추출 → 음성 텍스트 변환 → 청크 분할 → 병렬 요약

작성자 팁: LangGraph는 각 처리 단계를 "노드(node)"로 만들고,
노드들을 "엣지(edge)"로 연결해서 전체 흐름을 그래프처럼 관리하는 도구예요.
"""

# ─────────────────────────────────────────
# 라이브러리 임포트
# ─────────────────────────────────────────

from typing import TypedDict                        # 딕셔너리 키의 타입을 명시할 때 사용해요
from langgraph.graph import END, START, StateGraph  # 그래프의 시작/끝 노드, 그래프 빌더
from langgraph.types import Send                    # 병렬 작업을 각 워커에 분배할 때 사용
import subprocess                                   # ffmpeg 같은 외부 명령어를 실행할 때 필요해요
import os                                          # 환경변수(API 키 등)를 읽어올 때 사용
import base64                                      # 오디오 파일을 API에 전송하기 위해 바이트 → 문자열 변환
import textwrap                                    # 긴 텍스트를 일정 길이로 잘라줄 때 사용
from typing_extensions import Annotated            # 타입 힌트에 추가 정보(메타데이터)를 붙일 때 사용
import operator                                    # operator.add → 리스트를 합칠 때 쓰는 내장 연산자

# LangChain / Vertex AI / Google GenAI 관련 임포트
# 참고: init_chat_model은 현재 코드에서 직접 사용하지 않지만, 나중에 모델을 바꿀 때 편리해요
from langchain.chat_models import init_chat_model
from langchain_google_vertexai import ChatVertexAI  # Google Cloud의 Vertex AI 모델을 LangChain으로 쓸 때
from google import genai                           # Google의 공식 GenAI SDK (음성 → 텍스트 변환에 사용)


# ─────────────────────────────────────────
# LLM(대규모 언어 모델) 설정
# ─────────────────────────────────────────

# ChatVertexAI: Google Cloud Vertex AI에서 호스팅하는 Gemini 모델을 사용해요
# 텍스트 요약(summarize_chunk) 단계에서 이 LLM이 호출됩니다
llm = ChatVertexAI(
    model_name="gemini-2.5-flash-lite",             # 사용할 모델 이름
    project="ai-prompt-evaluator-489612",           # Google Cloud 프로젝트 ID
    location="us-central1",                         # 모델이 실행되는 데이터센터 위치
    max_tokens=500                                  # 응답 최대 길이 (너무 길면 잘림)
)


# ─────────────────────────────────────────
# State: 파이프라인 전체가 공유하는 데이터 구조
# ─────────────────────────────────────────

class State(TypedDict):
    """
    LangGraph 파이프라인 전체에서 공유되는 상태(State)입니다.
    각 노드가 이 State를 읽고 쓰면서 데이터가 흘러가요.

    생각해보면 마치 공유 메모지 같아요 ─
    한 단계가 값을 적으면, 다음 단계가 그 값을 읽을 수 있어요.
    """
    video_file: str                                 # 처리할 원본 영상 파일 경로 (예: "korea_2.mp4")
    audio_file: str                                 # 추출된 오디오 파일 경로 (예: "korea_2.mp3")
    transcription: str                             # Gemini가 변환해준 전체 텍스트(자막)
    # Annotated[list[str], operator.add] 의미:
    #   여러 워커가 동시에 결과를 돌려줄 때, 리스트를 덮어쓰지 않고 합쳐줘요 (append가 아닌 extend)
    summaries: Annotated[list[str], operator.add]  # 각 청크 요약 결과들이 모이는 리스트


# ─────────────────────────────────────────
# 노드 1: 영상에서 오디오 추출
# ─────────────────────────────────────────

def extract_audio(state: State):
    """
    ffmpeg를 이용해 mp4 영상에서 mp3 오디오를 추출합니다.
    재생 속도도 2배로 높여서 API 처리량을 줄여줘요 (비용 절감 효과도 있어요!).

    반환값: audio_file 경로를 State에 저장합니다.
    """
    # 파일 확장자를 mp4 → mp3 로 바꿔서 출력 파일 이름을 만들어요
    output_file = state["video_file"].replace("mp4", "mp3")

    # ffmpeg 명령어를 리스트 형태로 구성해요
    # -i: 입력 파일 지정
    # -filter:a "atempo=2.0": 오디오 속도를 2배로 높임 (2배 이상은 2단계로 나눠야 해요)
    # -y: 같은 이름의 파일이 있으면 덮어쓰기 허용
    command = [
        "ffmpeg",
        "-i",
        state["video_file"],
        "-filter:a",
        "atempo=2.0",
        "-y",
        output_file,
    ]

    # subprocess.run()으로 실제 터미널 명령어를 실행해요
    # ffmpeg -i korea_2.mp4 -filter:a atempo=2.0 -y korea_2.mp3
    subprocess.run(command)

    # 다음 노드에서 사용할 수 있도록 오디오 파일 경로를 State에 저장합니다
    return {
        "audio_file": output_file,
    }


# ─────────────────────────────────────────
# 노드 2: 오디오 → 텍스트 변환 (Transcription)
# ─────────────────────────────────────────

def transcribe_audio(state: State):
    """
    Google GenAI SDK를 사용해 mp3 오디오를 한국어 텍스트로 변환합니다.
    Gemini 모델에 오디오 파일을 직접 전달해서 음성 인식을 수행해요.

    ⚠️ 버그 수정: {"text", prompt} → {"text": prompt} (세트가 아닌 딕셔너리여야 해요!)
    """
    # 환경변수에서 Google API 키를 읽어옵니다
    # .env 파일에 GOOGLE_API_KEY=... 형태로 저장해두고, load_dotenv()를 먼저 호출해야 해요
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY가 없습니다. .env를 채우고 load_dotenv() 호출이 되었는지 확인하세요"
        )

    # GenAI 클라이언트를 API 키로 초기화합니다
    client = genai.Client(api_key=api_key)

    # 오디오 파일을 바이트로 읽고, base64 문자열로 인코딩합니다
    # API는 바이너리 파일을 직접 받지 않고, 문자열(base64) 형태로 전달받아요
    with open(state["audio_file"], "rb") as audio_file:
        audio_bytes = audio_file.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    # Gemini에게 전달할 지시문 (프롬프트)
    prompt = (
        "Transcribe the audio to Korean."
        "This country is the Republic of Korea"
        "Return only the transcription text."
    )

    # Gemini 모델 호출: 텍스트 지시문 + 오디오 데이터를 함께 전달합니다
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            {
                "role": "user",
                "parts": [
                    # ✅ 버그 수정: {"text", prompt} 는 Python set(집합) 문법이에요
                    #    올바른 딕셔너리 형태인 {"text": prompt} 로 수정했습니다
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/mpeg",   # mp3 파일의 MIME 타입
                            "data": audio_b64,          # base64로 인코딩된 오디오 데이터
                        }
                    },
                ],
            }
        ],
    )

    # 응답에서 텍스트를 안전하게 추출합니다
    # Gemini SDK 버전에 따라 응답 구조가 다를 수 있어서, 여러 방법으로 시도해요
    transcription = getattr(response, "text", None)
    if not transcription:
        try:
            # response.text가 없을 때 candidates 배열에서 직접 꺼내는 대안 방법
            transcription = response.candidates[0].content.parts[0].text
        except Exception:
            transcription = None

    # 위 두 방법 모두 실패하면 response 전체를 문자열로 변환합니다 (디버깅용)
    if not transcription:
        transcription = str(response)

    return {
        "transcription": transcription,
    }


# ─────────────────────────────────────────
# 조건부 엣지: 텍스트를 청크로 나눠 병렬 처리
# ─────────────────────────────────────────

def dispatch_summarizers(state: State):
    """
    긴 텍스트(전사본)를 500자씩 잘라서 각 청크를 summarize_chunk 노드로 보냅니다.
    LangGraph의 Send()를 사용하면 여러 노드에 동시에 데이터를 보낼 수 있어요 (팬아웃 패턴).

    예: 텍스트가 1500자라면 → 청크 3개를 동시에 처리 → 결과가 summaries에 합쳐져요
    """
    transcription = state["transcription"]
    chunks = []

    # textwrap.wrap: 텍스트를 단어 단위로 잘라서 최대 500자짜리 덩어리 리스트를 만들어요
    for i, chunk in enumerate(textwrap.wrap(transcription, 500)):
        chunks.append({
            "id": i + 1,       # 청크 번호 (1부터 시작)
            "chunk": chunk,    # 실제 텍스트 내용
        })

    # Send("노드이름", 데이터): 해당 노드를 데이터와 함께 실행해줘요
    # 여러 Send()를 리스트로 반환하면 동시(병렬)에 실행됩니다
    return [
        Send("summarize_chunk", chunk) for chunk in chunks
    ]


# ─────────────────────────────────────────
# 노드 3: 청크 요약
# ─────────────────────────────────────────

def summarize_chunk(chunk):
    """
    각 텍스트 청크를 LLM(Gemini)으로 요약합니다.
    dispatch_summarizers가 나눠준 청크 하나씩 이 함수가 독립적으로 처리해요.

    반환값: "[Chunk N] 요약 내용" 형식의 문자열이 summaries 리스트에 추가됩니다.
    """
    chunk_id = chunk["id"]     # 몇 번째 청크인지
    chunk = chunk["chunk"]     # 실제 텍스트 내용

    # LLM에게 한국어 요약 요청을 보냅니다
    response = llm.invoke(
        f"""
        아래 텍스트를 한국어로 간결하게 요약해 주세요.

        텍스트: {chunk}
        """
    )

    # 청크 번호를 앞에 붙여서 어떤 청크의 요약인지 구분하기 쉽게 만들어요
    summary = f"[Chunk {chunk_id}] {response.content}"

    # summaries는 Annotated[list[str], operator.add] 타입이므로,
    # 리스트 형태로 반환해야 여러 병렬 결과가 자동으로 합쳐져요
    return {
        "summaries": [summary],
    }


# ─────────────────────────────────────────
# LangGraph 파이프라인 구성
# ─────────────────────────────────────────

# StateGraph: 각 노드와 엣지를 등록해서 실행 흐름을 정의해요
graph_builder = StateGraph(State)

# 노드 등록: "이름" → 실행할 함수를 연결합니다
graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)

# 엣지 등록: 노드 실행 순서를 지정합니다
graph_builder.add_edge(START, "extract_audio")               # 시작 → 오디오 추출
graph_builder.add_edge("extract_audio", "transcribe_audio")  # 오디오 추출 → 텍스트 변환

# 조건부 엣지: transcribe_audio 다음에 dispatch_summarizers를 실행해서
# 결과에 따라 동적으로 summarize_chunk 노드를 여러 개 실행합니다
graph_builder.add_conditional_edges(
    "transcribe_audio",         # 이 노드가 끝난 후
    dispatch_summarizers,       # 이 함수가 다음 노드(들)을 결정
    ["summarize_chunk"]         # 가능한 다음 노드 목록 (라우팅 대상)
)

graph_builder.add_edge("summarize_chunk", END)               # 요약 완료 → 종료

# 그래프를 컴파일합니다 (실행 가능한 형태로 변환)
graph = graph_builder.compile()


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────

# graph.invoke(): 그래프를 실행해요. 초기 State에 영상 파일 경로만 넣어주면 돼요.
# 결과로 State 전체(summaries 포함)가 반환됩니다.
result = graph.invoke(
    {
        "video_file": "korea_2.mp4"  # 요약할 영상 파일 경로
    }
)

# 요약 결과 출력 (선택사항)
for summary in result.get("summaries", []):
    print(summary)
