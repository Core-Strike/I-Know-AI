"""
models.py
---------
FastAPI 요청/응답 Pydantic 스키마
"""

from typing import Any
from pydantic import BaseModel, Field


class AnalyzeResponse(BaseModel):
    """POST /analyze/{studentId} 응답"""

    studentId:    str        = Field(..., description="교육생 식별자")
    confused:     bool       = Field(..., description="혼란 여부 (GPT 판정)")
    confidence:   float      = Field(..., ge=0.0, le=1.0, description="FER 감정 최고 확률")
    emotion:      str        = Field(..., description="FER 최고 감정 레이블")
    gptReason:    str        = Field(..., alias="gpt_reason", description="GPT 판단 이유")
    faceFeatures: dict[str, Any] = Field(..., alias="face_features", description="FER + MediaPipe 원시 지표")

    class Config:
        populate_by_name = True


class ErrorResponse(BaseModel):
    detail: str
