FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 설치 (OpenCV, MediaPipe 의존성)
# requirements.txt 변경과 무관하게 캐시 재사용됨
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt만 먼저 복사 → 내용 변경 없으면 pip install 레이어 캐시 재사용
COPY requirements.txt .

# --no-cache-dir 제거: pip 캐시를 Docker 레이어에 보존해 재빌드 속도 향상
RUN pip install -r requirements.txt

# 소스 복사 (자주 바뀌는 파일은 마지막에)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
