#!/bin/bash
# 베이스 이미지 빌드 스크립트
# requirements.txt 변경 시에만 실행하면 됩니다

echo "=== confusion-api-base 이미지 빌드 시작 ==="
docker build -f Dockerfile.base -t confusion-api-base:latest .

if [ $? -eq 0 ]; then
  echo "=== 베이스 이미지 빌드 완료 ==="
else
  echo "=== 베이스 이미지 빌드 실패 ==="
  exit 1
fi
