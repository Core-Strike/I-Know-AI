# 베이스 이미지: 무거운 패키지가 미리 설치된 이미지
# 최초 1회 또는 requirements.txt 변경 시에만 재빌드
# docker build -f Dockerfile.base -t confusion-api-base . && docker push confusion-api-base
FROM confusion-api-base:latest

WORKDIR /app

# 소스코드만 복사 (빠름)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
