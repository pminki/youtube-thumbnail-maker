from itertools import starmap
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt, Command
from typing import TypedDict
import subprocess
import os
import json
import time
from langchain_google_vertexai import ChatVertexAI
from langchain_google_vertexai.vision_models import VertexAIImageGeneratorChat
import textwrap
from langchain.chat_models import init_chat_model
from typing_extensions import Annotated
import operator
import base64
from PIL import Image, ImageDraw, ImageFont           # Pillow 추가  
from google import genai

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

vertexai.init(project="ai-prompt-evaluator-489612", location="us-central1")

llm = ChatVertexAI(
  model_name="gemini-2.5-flash-lite",
  max_tokens=500
)

imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")  

class State(TypedDict):
  video_file: str
  audio_file: str
  transcription: str
  summaries: Annotated[list[str], operator.add]
  thumbnail_prompts: Annotated[list[str], operator.add]
  thumbnail_sketches: Annotated[list[str], operator.add]
  final_summary: str
  user_feedback: str
  chosen_thumbnail: int


def extract_audio(state: State):
  """
  ffmpeg를 이용해 mp4 영상에서 mp3 오디오를 추출합니다.
  재생 속도도 2배로 높여서 API 처리량을 줄여줘요 (비용 절감 효과도 있어요!)

  반환값: audio_file 경로를 State에 저장합니다.
  """
  
  # LangGraph 실행 입력에 video_file이 없으면 여기서 명확하게 안내합니다.
  # 기존 KeyError보다 원인 파악이 쉬운 에러 메시지를 제공합니다.
  video_file = state.get("video_file")
  if not video_file:
    raise ValueError(
      "입력 데이터에 'video_file'이 없습니다. "
      "예: {'video_file': 'korea_2.mp4'} 형태로 graph를 실행하세요."
    )

  # 파일 경로가 실제로 있는지도 먼저 확인합니다.
  if not os.path.exists(video_file):
    raise FileNotFoundError(
      f"입력 영상 파일을 찾을 수 없습니다: {video_file}"
    )

  # 파일 확장자를 mp4 -> mp3로 바꿔서 출력 파일 이름을 만들어요.
  output_file = video_file.replace("mp4", "mp3")

  # ffmpeg 명령어를 리스트 형태로 구성해요
  # -i: 입력 파일 지정
  # -filter:a "atempo=2.0": 오디오 속도를 2배로 높임 (2배 이상은 2단계로 나눠야 해요)
  # -y: 같은 이름의 파일이 있으면 덮어쓰기 허용
  command = [
    "ffmpeg",
    "-i",
    video_file,
    "-filter:a",
    "atempo=2.0",
    "-y",
    output_file,
  ]

  # subprocess.run()으로 실제 터미널 명령어를 실행해요.
  # ffmpeg -i korea_2.mp4 -filter:a atempo=2.0 -y korea_2.mp3
  subprocess.run(command, check=True)

  # 다음 노드에서 사용할 수 있도록 오디오 파일 경로를 State에 저장합니다.
  return {
    "audio_file": output_file
  }


# ─────────────────────────────────────────
# 노드 2: 오디오 → 텍스트 변환 (Transcription)
# ─────────────────────────────────────────

def transcribe_audio(state: State):
  """
  Google GenAI SDK를 사용해 mp3 오디오를 한국어 텍스트로 변홥합니다.
  Gemini 모델에 오디오 파일을 직접 전달해서 음성 인식을 수행해요.
  """

  api_key = os.environ.get("GOOGLE_API_KEY")
  if not api_key:
    raise ValueError(
      "GOOGLE_API_KEY가 없습니다. .env를 채우고 load_dotenv() 호출이 되었는지 확인하세요."
    )

  # GenAI 클라이언트를 API 키로 초기화합니다.
  client = genai.Client(api_key=api_key)

  # 오디오 파일을 바이트로 읽고, base64 문자열로 인코딩합니다.
  # API는 바이너리 파일을 직접 받지 않고, 문자열(base64) 형태로 전달받아요
  with open(state["audio_file"], "rb") as audio_file:
    audio_bytes = audio_file.read()

  audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

  # Gemini에게 전달할 지시문 (프롬프트)
  prompt = (
    "Trascribe the audio to Korean."
    "This country is the Republic of Korea"
    "Return only the transcription text"
  )

  # Gemini 모델 호출: 텍스트 지시문 + 오디오 데이터를 함께 전달합니다.
  response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=[
      {
        "role": "user",
        "parts": [
          {"text": prompt},
          {
            "inline_data": {
              "mime_type": "audio/mpeg",  # mp3 파일의 MIME 타입
              "data": audio_b64,          # base64로 인코딩된 오디오 데이터
            }
          },
        ],
      }
    ],
  )

  # 응답에서 텍스트를 안전하게 추출합니다.
  # Gemini SDK 버전에 따라 응답 구조가 다를 수 있어서, 여러 방법으로 시도해요
  transcription = getattr(response, "text", None)
  if not transcription:
    try:
      # response.text가 없을 때 candidates 배열에서 직접 꺼내는 대안 방법
      transcription = response.candidates[0].content.parts[0].text
    except Exception:
      transcription = None

  # 위 두 방법 모두 실해하면 response 전체를 문자열로 변환합니다. (디버깅용)
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
  # i와 잘려진 텍스트 조각(text_chunk)
  for i, text_chunk in enumerate(textwrap.wrap(transcription, 5000)):
    # chunk가 아닌 'chunks' 리스트에 append 합니다.
    chunks.append({
      "id": i + 1,    
      "chunk": text_chunk, 
    })

  # Send("노드이름", 데이터): 해당 노드를 데이터와 함께 실행해줘요.
  # 여러 Send()를 리스트로 반환하면 동시(병렬)에 실행됩니다.  
  return [
    Send("summarize_chunk", chunk_data) for chunk_data in chunks
  ]


# ─────────────────────────────────────────
# 노드 3: 청크 요약
# ─────────────────────────────────────────

def summarize_chunk(chunk):
  """
  각 텍스트 청크를 LLM(Gemini)으로 요약합니다.
  dispatch_summarizers가 나눠준 청크 하나씩 이 함수가 독립적으로 처리해요.

  반환값 "[Chunk N] 요약 내용" 형식의 문자열이 summaries 리스트에 추가됩니다.
  """
  chunk_id = chunk["id"]    # 몇 번째 청크인지
  text_chunk = chunk["chunk"]    # 실제 텍스트 내용

  # LLM에게 요약 요청을 보냅니다.
  response = llm.invoke(
    f"""
    아래 텍스트를 한국어로 간결하게 요약해 주세요.
    반드시 한국어로만 답변하세요.

    Text: {text_chunk}
    """
  )

  # 청크 번호를 앞에 붙여서 어떤 청크의 요약인지 구분하기 쉽게 만들어요.
  summary = f"[Chunk {chunk_id}] {response.content}"

  # summaries는 Annotated[list[str], operator.add] 타입이므로,
  # 리스트 형태로 반환해야 여러 병렬 결과가 자동으로 합쳐져요
  return {
    "summaries": [summary],
  }


def mega_summary(state: State):
  all_summaries = "\\n".join(state["summaries"])

  prompt = f"""
    You are given multiple summaries of different chunks from a video transcription.
    Please create a comprehensive final summary that combines all the key points.
    Individual summaries:
    {all_summaries}
  """
  response = llm.invoke(prompt)

  return {
    "final_summary": response.content,
  }
  

def dispatch_artists(state: State):
  return [
    Send(
      "generate_thumbnails",
      {
        "id": i,
        "summary": state["final_summary"],
      },
    )

    for i in [10, 11]
  ]



def generate_thumbnails(args):
  concept_id = args["id"]
  summary = args["summary"]

  # 1. 텍스트 프롬프트 생성 (gemini-2.5-flash-lite)
  # 1. LLM에게 이미지 프롬프트와 한글 타이틀을 '분리해서' JSON으로 받기
  prompt = f"""
    Based on this video summary, create a YouTube thumbnail concept.
    Return ONLY a valid JSON object with two keys:
    - "thumbnail_title": A catchy, short Korean title for the thumbnail (3-6 words).
    - "image_prompt": A prompt for an image generation model. 
      * IMPORTANT for image_prompt: Describe a scene with a clear focal subject and deliberate EMPTY/NEGATIVE SPACE (top, bottom, or side) for text placement later. DO NOT ask the image generator to draw text. Include dramatic lighting and vibrant colors.

    Summary: {summary}
  """

  prompt_response = llm.invoke(prompt)
  
  try:
    # LLM 응답이 JSON 텍스트라고 가정하고 파싱
    response_data = json.loads(prompt_response.content)
    korean_title = response_data.get("thumbnail_title", "위대한 대한민국")
    thumbnail_prompt = response_data.get("image_prompt", "")
  except Exception:
    # JSON 파싱 실패 시 대비책
    korean_title = "아시아의 중심이자 세계와 교류하는 나라, 대한민국"
    thumbnail_prompt = prompt_response.content

  # 2. Vertex AI Imagen SDK로 이미지 생성 (텍스트 없이 여백만 있는 이미지)
  max_retries = 3
  for attempt in range(max_retries):
      try:
          images = imagen_model.generate_images(
              prompt=thumbnail_prompt,
              number_of_images=1,
              aspect_ratio="16:9",
              safety_filter_level="block_few",
          )
          break
      except Exception as e: # ResourceExhausted 등
          if attempt < max_retries - 1:
              wait_seconds = 30 * (attempt + 1)
              print(f"⚠️ [{concept_id}번] 대기 중... {wait_seconds}초 후 재시도")
              time.sleep(wait_seconds)
          else:
              return {"thumbnail_prompts": [thumbnail_prompt], "thumbnail_sketches": []}

  # 3. 파일로 먼저 임시 저장 (Imagen 객체에서 PIL 객체로 변환하기 위함)
  temp_file_name = f"temp_concept_{concept_id}.png"
  images[0].save(location=temp_file_name, include_generation_parameters=False)

  # 4. Pillow(PIL)를 사용하여 한글 텍스트 합성
  img = Image.open(temp_file_name)
  draw = ImageDraw.Draw(img)
  
  # 폰트 설정 (시스템에 있는 한글 폰트 경로 지정 필요)
  # 예: 윈도우 "C:/Windows/Fonts/malgun.ttf", 맥 "/Library/Fonts/AppleGothic.ttf"
  font_path = "NanumGothicBold.ttf" 
  
  try:
      # 이미지 크기에 맞게 폰트 크기 조절 (예: 16:9 이미지 기준 80)
      font = ImageFont.truetype(font_path, 80)
  except IOError:
      font = ImageFont.load_default()
      print("⚠️ 폰트 파일을 찾을 수 없어 기본 폰트를 사용합니다. 한글이 깨질 수 있습니다.")

  # 텍스트 위치 및 색상 설정 (유튜브 썸네일 스타일의 텍스트 테두리 효과)
  text_position = (50, 50) # 좌측 상단 여백 (조정 필요)
  text_color = (255, 255, 255) # 흰색 텍스트
  outline_color = (0, 0, 0)    # 검은색 테두리
  outline_width = 3

  # 테두리 그리기
  for x_offset in range(-outline_width, outline_width + 1):
    for y_offset in range(-outline_width, outline_width + 1):
      draw.text((text_position[0] + x_offset, text_position[1] + y_offset), 
                korean_title, font=font, fill=outline_color)
  
  # 실제 텍스트 그리기
  draw.text(text_position, korean_title, font=font, fill=text_color)

  # 최종 이미지 저장
  final_file_name = f"thumbnail_concept_{concept_id}.png"
  img.save(final_file_name)
  print(f"✅ {final_file_name} 저장 및 텍스트 합성 완료!")

  # 5. State 반환용 base64 변환
  with open(final_file_name, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")
  
  image_base64_url = f"data:image/png;base64,{image_b64}"
  
  return {
    "thumbnail_prompts": [thumbnail_prompt],
    "thumbnail_sketches": [image_base64_url],
    "thumbnail_korean_title": korean_title # 참조용으로 State에 추가
  }



def human_feedback(state: State):
  """
  사용자에게 최종 선택 썸네일 번호와 수정 피드백을 받아,
  다음 노드(generate_hd_thumbnail)에서 사용할 데이터로 정리합니다.
  """
  answer = interrupt(
    {
      "chosen_thumbnail": "Which thumbnail do you like the most? (index)",
      "feedback": "Provide any feedback or changes you'd like for the final thumbnail.",
    }
  )

  # interrupt 응답은 환경/호출 방식에 따라 dict 또는 str로 들어올 수 있습니다.
  # - dict: {"feedback": "...", "chosen_thumbnail": 1}
  # - str : "피드백 텍스트" 또는 JSON 문자열
  if isinstance(answer, dict):
    parsed_answer = answer
  elif isinstance(answer, str):
    text = answer.strip()
    try:
      # 문자열이 JSON이면 파싱해서 사용
      parsed_answer = json.loads(text)
      if not isinstance(parsed_answer, dict):
        parsed_answer = {"feedback": text}
    except Exception:
      # 일반 문자열이면 feedback으로 간주
      parsed_answer = {"feedback": text}
  else:
    parsed_answer = {}

  # interrupt 응답에서 직접 읽어야 합니다.
  # (기존처럼 state["user_feedback"]를 바로 읽으면 KeyError가 날 수 있음)
  user_feedback = parsed_answer.get("feedback", "")
  chosen_thumbnail = parsed_answer.get("chosen_thumbnail")

  thumbnail_prompts = state.get("thumbnail_prompts", [])

  # 사용자가 선택한 번호를 실제 프롬프트 문자열로 변환합니다.
  chosen_prompt = None
  if isinstance(chosen_thumbnail, int):
    if 0 <= chosen_thumbnail < len(thumbnail_prompts):
      chosen_prompt = thumbnail_prompts[chosen_thumbnail]
  elif isinstance(chosen_thumbnail, str) and chosen_thumbnail.strip().isdigit():
    idx = int(chosen_thumbnail.strip())
    if 0 <= idx < len(thumbnail_prompts):
      chosen_prompt = thumbnail_prompts[idx]

  # 번호 매핑 실패 시, 기존 state 값이나 첫 번째 프롬프트로 안전하게 대체
  if chosen_prompt is None:
    chosen_prompt = state.get("chosen_prompt")
  if chosen_prompt is None and thumbnail_prompts:
    chosen_prompt = thumbnail_prompts[0]

  if chosen_prompt is None:
    raise ValueError("선택 가능한 thumbnail_prompts가 없습니다. generate_thumbnails 결과를 먼저 확인하세요.")

  return {
    "user_feedback": user_feedback,
    "chosen_prompt": chosen_prompt,
  }


def generate_hd_thumbnail(state: State):
  # 우선 직접 전달된 chosen_prompt를 사용하고,
  # 없으면 chosen_thumbnail + thumbnail_prompts로 복구합니다.
  chosen_prompt = state.get("chosen_prompt")

  if not chosen_prompt:
    thumbnail_prompts = state.get("thumbnail_prompts", [])
    chosen_thumbnail = state.get("chosen_thumbnail")

    idx = None
    if isinstance(chosen_thumbnail, int):
      idx = chosen_thumbnail
    elif isinstance(chosen_thumbnail, str) and chosen_thumbnail.strip().isdigit():
      idx = int(chosen_thumbnail.strip())

    if idx is not None and 0 <= idx < len(thumbnail_prompts):
      chosen_prompt = thumbnail_prompts[idx]

  # 그래도 없으면 첫 번째 프롬프트로 fallback
  if not chosen_prompt:
    thumbnail_prompts = state.get("thumbnail_prompts", [])
    if thumbnail_prompts:
      chosen_prompt = thumbnail_prompts[0]

  if not chosen_prompt:
    raise ValueError("chosen_prompt를 찾을 수 없습니다. human_feedback resume 데이터와 thumbnail_prompts를 확인하세요.")

  user_feedback = state.get("user_feedback", "")

  prompt = f"""
    You are a professional YouTube thumbnail designer. Take this original thumbnail prompt and create an enhanced version that incorporates the user's specific feedback.

    ORIGINAL PROMPT:
    {chosen_prompt}

    USER FEEDBACK TO INCORPORATE:
    {user_feedback}

    Create an enhanced prompt that:
        1. Maintains the core concept from the original prompt
        2. Specifically addresses and implements the user's feedback requests
        3. Adds professional YouTube thumbnail specifications:
            - High contrast and bold visual elements
            - Clear focal points that draw the eye
            - Professional lighting and composition
            - Optimal text placement and readability with generous padding from edges
            - Colors that pop and grab attention
            - Elements that work well at small thumbnail sizes
            - IMPORTANT: Always ensure adequate white space/padding between any text and the image borders
    """
  response = llm.invoke(prompt)

  final_thumbnail_prompt = response.content

  # OpenAI 이미지 API 대신 Vertex AI Imagen으로 최종 썸네일을 생성합니다.
  # 429(ResourceExhausted) 대응: 지수 백오프 재시도
  max_retries = 5
  images = None

  for attempt in range(max_retries):
    try:
      images = imagen_model.generate_images(
          prompt=final_thumbnail_prompt,
          number_of_images=1,
          aspect_ratio="16:9",
          safety_filter_level="block_few",
      )
      break
    except Exception as e:
      err_text = str(e)
      is_quota_error = ("ResourceExhausted" in err_text) or ("429" in err_text) or ("Quota exceeded" in err_text)

      # 쿼터 초과 계열 오류는 대기 후 재시도, 그 외 오류는 바로 재발생
      if not is_quota_error:
        raise

      if attempt < max_retries - 1:
        wait_seconds = min(180, 20 * (2 ** attempt))
        print(f"⚠️ Imagen 쿼터 초과(429). {wait_seconds}초 대기 후 재시도합니다... ({attempt + 1}/{max_retries})")
        time.sleep(wait_seconds)
      else:
        # 최종 실패 시 명확한 안내 메시지로 종료
        raise RuntimeError(
          "Imagen 호출이 쿼터(429) 초과로 반복 실패했습니다. "
          "잠시 후 다시 시도하거나 GCP Vertex AI quota 증설을 진행하세요."
        ) from e

  # Imagen 결과를 파일로 저장합니다.
  images[0].save(location="thumbnail_final.jpg", include_generation_parameters=False)

  with open("thumbnail_final.jpg", "rb") as file:
    image_bytes = file.read()

  image_b64 = base64.b64encode(image_bytes).decode("utf-8")

  return {
    "final_thumbnail_prompt": final_thumbnail_prompt,
    "final_thumbnail": f"data:image/jpeg;base64,{image_b64}",
  }
 

graph_builder = StateGraph(State)

graph_builder.add_node("extract_audio", extract_audio)
graph_builder.add_node("transcribe_audio", transcribe_audio)
graph_builder.add_node("summarize_chunk", summarize_chunk)
graph_builder.add_node("mega_summary", mega_summary)
graph_builder.add_node("generate_thumbnails", generate_thumbnails)
graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("generate_hd_thumbnail", generate_hd_thumbnail)

graph_builder.add_edge(START, "extract_audio")
graph_builder.add_edge("extract_audio", "transcribe_audio")
graph_builder.add_conditional_edges(
    "transcribe_audio", dispatch_summarizers, ["summarize_chunk"]
)
graph_builder.add_edge("summarize_chunk", "mega_summary")
graph_builder.add_conditional_edges(
    "mega_summary", dispatch_artists, ["generate_thumbnails"]
)
graph_builder.add_edge("generate_thumbnails", "human_feedback")
graph_builder.add_edge("human_feedback", "generate_hd_thumbnail")
graph_builder.add_edge("generate_hd_thumbnail", END)

graph = graph_builder.compile(name="mr_thumbs") 