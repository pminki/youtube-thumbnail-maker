import yt_dlp

# 다운로드할 유튜브 URL입니다.
# (원하는 영상 주소로 바꾸면 됩니다.)
url="https://www.youtube.com/watch?v=FS_-rSfc60E"

# yt-dlp의 동작을 결정하는 옵션 묶음입니다.
# 초보자 기준으로: "어떤 포맷으로 받을지", "어디에 저장할지",
# "다운로드 후 어떤 변환을 할지"를 여기서 지정합니다.
ydl_opts = {
  # 가능한 한 '영상+음성'을 가장 좋은 품질로 받도록 시도합니다.
  # bestvideo_bestaudio: 영상과 음성을 따로 최고 품질로 구한 뒤
  # merge(best) : 필요하면 합치거나 가장 적절한 조합을 선택합니다.
  "format": "bestvideo_bestaudio/best",

  # 다운로드 파일 저장 경로/이름 템플릿입니다.
  # %(ext)s는 다운로드 결과의 확장자(ext)로 자동 치환됩니다.
  # 예: temp.mp4 / temp.webm 같은 식으로 저장될 수 있습니다.
  "outtmpl": "./temp.%(ext)s",

  # (비디오+오디오를) 합칠 때 최종적으로 만들 형식을 지정합니다.
  # 여기서는 mp4로 합치길 원한다는 뜻입니다.
  "merge_output_format": "mp4",

  # postprocessors는 다운로드 후 추가 작업(후처리)을 하는 단계입니다.
  # 예: 특정 포맷으로 변환, 합치기 등.
  "postprocessors": [
    {
      # ffmpeg를 이용해 비디오를 원하는 포맷으로 변환하는 처리기입니다.
      "key": "FFmpegVideoConvertor",
      # 최종적으로 원하는 포맷(preferedformat)입니다.
      # NOTE: 이 필드 이름이 'preferedformat'으로 되어 있는데,
      # yt-dlp 버전에 따라 철자에 민감할 수 있어요.
      # 현재 코드는 그대로 두되, 만약 동작이 기대와 다르면
      # yt-dlp 문서에서 해당 옵션명을 확인해 보세요.
      "preferedformat": "mp4",
    }
  ],
}

# YoutubeDL는 "다운로드 엔진" 객체라고 생각하면 됩니다.
# with 구문을 쓰면 작업이 끝난 뒤 자원(핸들 등)을 안전하게 정리합니다.
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
  # extract_info(..., download=True)는
  # 1) 영상 메타데이터를 가져오고
  # 2) 실제로 다운로드까지 수행합니다.
  info = ydl.extract_info(url, download=True)

  # 영상 제목을 가져옵니다.
  # 만약 title이 없으면 기본값 "Unknown"을 사용합니다.
  title = info.get("title", "Unknown")

  # 다운로드가 끝났음을 콘솔에 출력합니다.
  print(f"{title} 다운로드 완료")