"""
main.py
-------
혼란 감지 FastAPI 서버

실행:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

엔드포인트:
    POST /analyze/{student_id}
        - multipart/form-data 로 이미지 파일 전송
        - confused: true/false 응답
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()   # .env 파일에서 OPENAI_API_KEY 등 로드

from fastapi import FastAPI, File, UploadFile, HTTPException, Path
from fastapi.middleware.cors import CORSMiddleware

from analyzer import FaceAnalyzer
from models import AnalyzeResponse, SummarizeRequest, SummarizeResponse, ErrorResponse

# ── 로깅 설정 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 앱 수명 주기: 분석기 싱글톤 초기화 ──────────────────────────────────────
analyzer: FaceAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("FaceAnalyzer 로딩 중...")
    analyzer = FaceAnalyzer()
    logger.info("FaceAnalyzer 준비 완료")
    yield
    logger.info("서버 종료")


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Confusion Detection API",
    description="교육생 웹캠 프레임을 받아 혼란 여부를 반환합니다.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 운영 환경에서는 Spring/프론트 도메인으로 좁히세요
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@app.post(
    "/analyze/{student_id}",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="교육생 얼굴 이미지 분석",
    description=(
        "10초마다 교육생 브라우저에서 캡처한 웹캠 프레임을 받아 "
        "MediaPipe + FER 로 혼란 여부를 판단합니다."
    ),
)
async def analyze(
    student_id: str = Path(..., description="교육생 식별자 (Spring과 동일 ID)"),
    file: UploadFile = File(..., description="웹캠 캡처 이미지 (JPEG 또는 PNG)"),
):
    # ── 파일 유효성 검사 ──────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 이미지 형식: {file.content_type}. "
                   "JPEG / PNG / WebP 만 허용됩니다.",
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    # 10 MB 상한
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="파일 크기가 10 MB를 초과합니다.")

    # ── 분석 ─────────────────────────────────────────────────────────────────
    result = analyzer.analyze(image_bytes)

    logger.info(
        "student=%s  confused=%s  emotion=%s  reason=%s",
        student_id,
        result["confused"],
        result["emotion"],
        result["gpt_reason"],
    )

    return AnalyzeResponse(
        studentId=student_id,
        confused=result["confused"],
        confidence=result["confidence"],
        emotion=result["emotion"],
        gpt_reason=result["gpt_reason"],
        face_features=result["face_features"],
    )


@app.post(
    "/ai-api/summarize",
    response_model=SummarizeResponse,
    responses={400: {"model": ErrorResponse}},
    summary="강의 텍스트 AI 요약",
    description=(
        "혼란 이벤트 발생 후 2분간 녹음된 강의 STT 텍스트를 받아 "
        "GPT로 요약문을 생성합니다. Spring에서 위임 호출합니다."
    ),
)
async def summarize(body: SummarizeRequest):
    if not body.audioText.strip():
        raise HTTPException(status_code=400, detail="audioText가 비어있습니다.")

    summary = analyzer.summarize(body.audioText)

    logger.info(
        "summarize  alertId=%s  sessionId=%s  textLen=%d",
        body.alertId,
        body.sessionId,
        len(body.audioText),
    )

    return SummarizeResponse(
        alertId=body.alertId,
        summary=summary,
        audioText=body.audioText,
    )


@app.get("/health", summary="헬스 체크")
async def health():
    return {"status": "ok", "analyzer_ready": analyzer is not None}
