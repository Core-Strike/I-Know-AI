"""
FastAPI application for confusion analysis and lecture summarization.
"""

import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analyzer import FaceAnalyzer
from models import AnalyzeResponse, ErrorResponse, SummarizeRequest, SummarizeResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

analyzer: FaceAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("Loading FaceAnalyzer...")
    analyzer = FaceAnalyzer()
    logger.info("FaceAnalyzer ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Confusion Detection API",
    description="Analyze student frames and summarize lecture transcripts.",
    version="1.0.0",
    lifespan=lifespan,
)

router = APIRouter(prefix="/ai-api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET", "DELETE"],
    allow_headers=["*"],
)


@router.post(
    "/analyze/{student_id}",
    response_model=AnalyzeResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Analyze a student frame",
)
async def analyze(
    student_id: str = Path(..., description="Student identifier"),
    file: UploadFile = File(..., description="Captured student frame"),
):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    image_bytes = await file.read()

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File exceeds the 10 MB limit.")

    result = analyzer.analyze(image_bytes)

    logger.info(
        "student=%s confused=%s emotion=%s",
        student_id,
        result["confused"],
        result["emotion"],
    )

    return AnalyzeResponse(
        studentId=student_id,
        confused=result["confused"],
        confidence=result["confidence"],
        emotion=result["emotion"],
        gpt_reason=result["gpt_reason"],
        face_features=result["face_features"],
    )


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    responses={400: {"model": ErrorResponse}},
    summary="Summarize a lecture transcript",
)
async def summarize(body: SummarizeRequest):
    if not body.audioText.strip():
        raise HTTPException(status_code=400, detail="audioText is empty.")

    result = analyzer.summarize(body.audioText)
    logger.info("summarize textLen=%d", len(body.audioText))

    return SummarizeResponse(
        summary=result["summary"],
        recommendedConcept=result["recommendedConcept"],
    )


@router.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "analyzer_ready": analyzer is not None}


app.include_router(router)
