"""
analyzer.py
-----------
MediaPipe Face Landmarker (Tasks API, 0.10.x 호환) + FER → 원시 데이터 수집
OpenAI GPT → 이해도(혼란 여부) 최종 판정

흐름:
  1. FER      : 7개 감정 확률 점수 추출
  2. MediaPipe: 눈썹-눈 거리 비율, 눈 개폐율(EAR), 고개 기울기 추출
  3. 위 값들을 GPT에 전달 → confused: true/false 판정
"""

import cv2
import json
import logging
import math
import os
import urllib.request

import mediapipe as mp
import numpy as np
from fer import FER
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── MediaPipe 모델 설정 ───────────────────────────────────────────────────────
# Windows: 한글 경로에서 MediaPipe C++이 파일을 못 읽는 문제 → 홈 디렉터리(영문) 사용
# Linux(배포): 코드와 같은 디렉터리 사용
import platform
if platform.system() == "Windows":
    MODEL_PATH = os.path.join(os.path.expanduser("~"), "face_landmarker.task")
else:
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── OpenAI 설정 ───────────────────────────────────────────────────────────────
GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
GPT_SYSTEM_PROMPT = """\
당신은 교육 현장에서 교육생의 표정 데이터를 분석해 이해도를 판단하는 전문가입니다.
아래 얼굴 표정 지표를 보고, 이 교육생이 현재 강의 내용을 이해하지 못해 혼란스러운지 종합적으로 판단하세요.

━━ 데이터 수집 도구 및 신뢰도 ━━

[FER (Facial Expression Recognition)]
딥러닝 기반 표정 인식 라이브러리로, 정적 이미지에서 7가지 감정을 분류합니다.
- 웹캠에서 10초마다 캡처한 단일 프레임을 분석한 결과입니다
- 조명, 각도, 해상도에 따라 정확도가 달라질 수 있습니다
- 개인마다 표정 표현 방식이 다를 수 있어 절대적인 수치보다 상대적 분포를 참고하세요
- 무표정(neutral)이 높다고 해서 반드시 이해한 상태는 아닙니다 (집중 또는 무관심일 수도 있음)

[MediaPipe Face Landmarker]
Google이 개발한 얼굴 랜드마크 감지 라이브러리로, 478개의 얼굴 특징점 좌표를 추출합니다.
- 감정을 직접 감지하는 것이 아니라 얼굴 기하학적 수치를 측정합니다
- 정면 얼굴 기준으로 설계되어 있어 고개가 많이 돌아간 경우 수치 신뢰도가 낮아집니다
- 조명이 어둡거나 얼굴이 부분적으로 가려지면 랜드마크 정확도가 떨어질 수 있습니다

━━ 각 지표의 의미 ━━

[FER 감정 점수]
모든 감정 점수의 합은 1.0이며, 각 값은 해당 감정이 표정에서 감지된 확률입니다.
• happy    : 밝고 편안한 표정
• neutral  : 무표정. 집중하거나 수동적으로 듣는 상태일 수 있음
• surprise : 놀람. 예상치 못한 내용에 반응하는 것으로, 혼란과 구분 필요
• fear     : 두려움 또는 당혹감. 내용이 어렵거나 압도될 때 나타날 수 있음
• sad      : 슬픔 또는 무기력감. 이해를 포기하거나 좌절하는 상태와 연관될 수 있음
• angry    : 짜증 또는 답답함. 이해가 안 될 때 발생하는 내적 저항 표현일 수 있음
• disgust  : 불쾌감. 내용이 맞지 않거나 거부감을 느낄 때 나타날 수 있음

[MediaPipe 얼굴 랜드마크 지표]
• brow_eye_ratio : 눈썹과 눈 사이의 수직 거리 비율. 낮을수록 눈썹이 아래로 내려와 찡그린 상태
• ear            : 눈의 세로/가로 비율(Eye Aspect Ratio). 낮을수록 눈이 감겨있는 상태, 높을수록 눈을 크게 뜬 상태
• head_tilt_deg  : 고개 기울기 각도(도 단위). 클수록 고개를 옆으로 기울이고 있는 상태
• face_detected  : 얼굴이 카메라에 감지되었는지 여부. false이면 자리를 비웠거나 카메라에서 벗어난 상태

━━ 응답 형식 ━━
반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이:
{"confused": true, "reason": "한 문장 이유"}
"""

# ── 랜드마크 인덱스 (MediaPipe 478-point Face Mesh 기준) ─────────────────────
LEFT_BROW_TOP  = [70, 63, 105, 66, 107]
LEFT_EYE_TOP   = [33, 160, 158, 133]
LEFT_EYE_BOT   = [145, 153, 154, 155]
RIGHT_BROW_TOP = [336, 296, 334, 293, 300]
RIGHT_EYE_TOP  = [362, 385, 387, 263]
RIGHT_EYE_BOT  = [374, 380, 381, 382]


def _ensure_model():
    """face_landmarker.task 모델 파일이 없으면 자동 다운로드"""
    if not os.path.exists(MODEL_PATH):
        logger.info("face_landmarker.task 모델 다운로드 중... (약 5 MB)")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("모델 다운로드 완료: %s", MODEL_PATH)


class FaceAnalyzer:
    """이미지 바이트를 받아 GPT 기반 이해도 판정 결과를 반환하는 분석기"""

    def __init__(self):
        # FER (감정 추출)
        self._fer = FER(mtcnn=False)

        # MediaPipe Face Landmarker (Tasks API)
        _ensure_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        # OpenAI 클라이언트
        self._gpt = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        logger.info("FaceAnalyzer 초기화 완료 (GPT 모델: %s)", GPT_MODEL)

    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, image_bytes: bytes) -> dict:
        img = self._decode_image(image_bytes)
        if img is None:
            return self._fallback("이미지 디코딩 실패")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = self._extract_features(rgb, img.shape)
        return self._gpt_judge(features)

    # ── 원시 데이터 추출 ──────────────────────────────────────────────────────

    def _extract_features(self, rgb: np.ndarray, shape: tuple) -> dict:
        features = {
            "face_detected":  False,
            "emotions":       {},
            "top_emotion":    "unknown",
            "confidence":     0.0,
            "brow_eye_ratio": None,
            "ear":            None,
            "head_tilt_deg":  None,
        }

        # ── FER ──
        try:
            fer_results = self._fer.detect_emotions(rgb)
            if fer_results:
                face = max(fer_results, key=lambda r: r["box"][2] * r["box"][3])
                features["face_detected"] = True
                features["emotions"]      = {k: round(v, 3) for k, v in face["emotions"].items()}
                features["top_emotion"]   = max(face["emotions"], key=face["emotions"].get)
                features["confidence"]    = round(max(face["emotions"].values()), 3)
        except Exception as e:
            logger.warning("FER 오류: %s", e)

        # ── MediaPipe (Tasks API) ──
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = self._landmarker.detect(mp_image)

            if result.face_landmarks:
                features["face_detected"] = True
                lm = result.face_landmarks[0]   # List[NormalizedLandmark]
                h, w = shape[:2]

                features["brow_eye_ratio"] = round(self._brow_eye_ratio(lm), 4)
                features["ear"]            = round(self._eye_aspect_ratio(lm), 4)
                features["head_tilt_deg"]  = round(self._head_tilt(lm, w, h), 2)
        except Exception as e:
            logger.warning("MediaPipe 오류: %s", e)

        return features

    # ── 지표 계산 ─────────────────────────────────────────────────────────────

    def _brow_eye_ratio(self, lm) -> float:
        """눈썹 상단 ↔ 눈 상단 수직 거리 비율 (작을수록 찡그림)"""
        left_gap  = (np.mean([lm[i].y for i in LEFT_EYE_TOP])
                     - np.mean([lm[i].y for i in LEFT_BROW_TOP]))
        right_gap = (np.mean([lm[i].y for i in RIGHT_EYE_TOP])
                     - np.mean([lm[i].y for i in RIGHT_BROW_TOP]))
        return float((left_gap + right_gap) / 2)

    def _eye_aspect_ratio(self, lm) -> float:
        """Eye Aspect Ratio — 눈 세로/가로 비율"""
        def ear(top_ids, bot_ids, corner_l, corner_r):
            top   = np.mean([lm[i].y for i in top_ids])
            bot   = np.mean([lm[i].y for i in bot_ids])
            width = math.hypot(lm[corner_l].x - lm[corner_r].x,
                               lm[corner_l].y - lm[corner_r].y)
            return (bot - top) / (width + 1e-6)

        l = ear(LEFT_EYE_TOP,  LEFT_EYE_BOT,  33,  133)
        r = ear(RIGHT_EYE_TOP, RIGHT_EYE_BOT, 362, 263)
        return float((l + r) / 2)

    def _head_tilt(self, lm, w: int, h: int) -> float:
        """좌우 눈 중심 y좌표 차이로 고개 기울기 추정 (도 단위, 절댓값)"""
        left_cx  = np.mean([lm[i].x * w for i in LEFT_EYE_TOP])
        left_cy  = np.mean([lm[i].y * h for i in LEFT_EYE_TOP])
        right_cx = np.mean([lm[i].x * w for i in RIGHT_EYE_TOP])
        right_cy = np.mean([lm[i].y * h for i in RIGHT_EYE_TOP])
        return float(abs(math.degrees(math.atan2(right_cy - left_cy,
                                                 right_cx - left_cx + 1e-6))))

    # ── GPT 판정 ──────────────────────────────────────────────────────────────

    def _gpt_judge(self, features: dict) -> dict:
        user_content = (
            f"얼굴 표정 지표:\n"
            f"{json.dumps(features, ensure_ascii=False, indent=2)}"
        )
        confused, gpt_reason = False, ""
        try:
            response = self._gpt.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                temperature=0,
                max_completion_tokens=100,
            )
            raw    = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            confused   = bool(parsed.get("confused", False))
            gpt_reason = parsed.get("reason", "")
        except json.JSONDecodeError:
            logger.error("GPT 응답 파싱 실패: %s", raw)
            gpt_reason = "GPT 응답 파싱 실패"
        except Exception as e:
            logger.error("GPT 호출 오류: %s", e)
            gpt_reason = f"GPT 호출 오류: {e}"

        logger.info("confused=%s | reason=%s", confused, gpt_reason)

        return {
            "confused":      confused,
            "confidence":    features["confidence"],
            "emotion":       features["top_emotion"],
            "gpt_reason":    gpt_reason,
            "face_features": features,
        }

    # ── 강의 텍스트 요약 ──────────────────────────────────────────────────────

    def summarize(self, audio_text: str) -> str:
        """
        혼란 이벤트 직후 2분간 녹음된 강의 STT 텍스트를 GPT로 요약합니다.

        Parameters
        ----------
        audio_text : str
            STT 변환된 강의 원문 텍스트

        Returns
        -------
        str
            GPT가 생성한 요약문
        """
        system_prompt = (
            "당신은 교육 현장의 강의 내용을 분석하는 전문가입니다.\n"
            "아래는 교육생이 혼란을 느낀 시점 직후 2분간 녹음된 강의 텍스트입니다.\n"
            "강사가 무엇을 설명하고 있었는지 핵심 내용을 3~5문장으로 요약해주세요.\n"
            "교육생이 어느 부분에서 어려움을 느꼈을지 추정하여 마지막에 한 문장으로 덧붙여주세요.\n"
            "응답은 한국어로 작성하세요."
        )
        try:
            response = self._gpt.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"강의 텍스트:\n{audio_text}"},
                ],
                temperature=0.3,
                max_completion_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("요약 GPT 호출 오류: %s", e)
            return f"요약 생성 실패: {e}"

    # ── 유틸 ─────────────────────────────────────────────────────────────────

    def _decode_image(self, image_bytes: bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _fallback(reason: str) -> dict:
        logger.debug("분석 불가: %s", reason)
        return {
            "confused":      False,
            "confidence":    0.0,
            "emotion":       "unknown",
            "gpt_reason":    reason,
            "face_features": {"face_detected": False},
        }
