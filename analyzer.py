"""
MediaPipe + FER based face analyzer and lecture summarizer.
"""

import cv2
import json
import logging
import math
import os
import platform
import re
import urllib.request

import mediapipe as mp
import numpy as np
from fer import FER
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from openai import OpenAI

logger = logging.getLogger(__name__)

if platform.system() == "Windows":
    MODEL_PATH = os.path.join(os.path.expanduser("~"), "face_landmarker.task")
else:
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

GPT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
GPT_SYSTEM_PROMPT = """\
당신은 교육 현장에서 교육생의 표정 데이터를 분석해 이해 여부를 판단하는 전문가입니다.
아래 얼굴 표정 지표를 보고, 이 교육생이 현재 강의 내용을 이해하지 못했는지 종합적으로 판단하세요.

`confused`는 이해하지 못한 상태 여부를 뜻합니다.
`confidence`는 감정 confidence가 아니라, 이해하지 못한 정도를 0부터 1 사이 값으로 나타냅니다.

반드시 아래 JSON 형식으로만 응답하세요.
{"confused": true, "confidence": 0.82, "reason": "학생이 현재 내용을 이해하지 못한 것으로 보입니다."}
"""

LEFT_BROW_TOP = [70, 63, 105, 66, 107]
LEFT_EYE_TOP = [33, 160, 158, 133]
LEFT_EYE_BOT = [145, 153, 154, 155]
RIGHT_BROW_TOP = [336, 296, 334, 293, 300]
RIGHT_EYE_TOP = [362, 385, 387, 263]
RIGHT_EYE_BOT = [374, 380, 381, 382]


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("Model downloaded to %s", MODEL_PATH)


class FaceAnalyzer:
    def __init__(self):
        self._fer = FER(mtcnn=False)

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
        self._gpt = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        logger.info("FaceAnalyzer initialized (GPT model: %s)", GPT_MODEL)

    def analyze(self, image_bytes: bytes) -> dict:
        img = self._decode_image(image_bytes)
        if img is None:
            return self._fallback("이미지 디코딩 실패")

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = self._extract_features(rgb, img.shape)
        return self._gpt_judge(features)

    def _extract_features(self, rgb: np.ndarray, shape: tuple) -> dict:
        features = {
            "face_detected": False,
            "emotions": {},
            "top_emotion": "unknown",
            "confidence": 0.0,
            "brow_eye_ratio": None,
            "ear": None,
            "head_tilt_deg": None,
        }

        try:
            fer_results = self._fer.detect_emotions(rgb)
            if fer_results:
                face = max(fer_results, key=lambda r: r["box"][2] * r["box"][3])
                features["face_detected"] = True
                features["emotions"] = {k: round(v, 3) for k, v in face["emotions"].items()}
                features["top_emotion"] = max(face["emotions"], key=face["emotions"].get)
                features["confidence"] = round(max(face["emotions"].values()), 3)
        except Exception as e:
            logger.warning("FER error: %s", e)

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect(mp_image)

            if result.face_landmarks:
                features["face_detected"] = True
                lm = result.face_landmarks[0]
                h, w = shape[:2]

                features["brow_eye_ratio"] = round(self._brow_eye_ratio(lm), 4)
                features["ear"] = round(self._eye_aspect_ratio(lm), 4)
                features["head_tilt_deg"] = round(self._head_tilt(lm, w, h), 2)
        except Exception as e:
            logger.warning("MediaPipe error: %s", e)

        return features

    def _brow_eye_ratio(self, lm) -> float:
        left_gap = np.mean([lm[i].y for i in LEFT_EYE_TOP]) - np.mean([lm[i].y for i in LEFT_BROW_TOP])
        right_gap = np.mean([lm[i].y for i in RIGHT_EYE_TOP]) - np.mean([lm[i].y for i in RIGHT_BROW_TOP])
        return float((left_gap + right_gap) / 2)

    def _eye_aspect_ratio(self, lm) -> float:
        def ear(top_ids, bot_ids, corner_l, corner_r):
            top = np.mean([lm[i].y for i in top_ids])
            bot = np.mean([lm[i].y for i in bot_ids])
            width = math.hypot(
                lm[corner_l].x - lm[corner_r].x,
                lm[corner_l].y - lm[corner_r].y,
            )
            return (bot - top) / (width + 1e-6)

        left = ear(LEFT_EYE_TOP, LEFT_EYE_BOT, 33, 133)
        right = ear(RIGHT_EYE_TOP, RIGHT_EYE_BOT, 362, 263)
        return float((left + right) / 2)

    def _head_tilt(self, lm, w: int, h: int) -> float:
        left_cx = np.mean([lm[i].x * w for i in LEFT_EYE_TOP])
        left_cy = np.mean([lm[i].y * h for i in LEFT_EYE_TOP])
        right_cx = np.mean([lm[i].x * w for i in RIGHT_EYE_TOP])
        right_cy = np.mean([lm[i].y * h for i in RIGHT_EYE_TOP])
        return float(abs(math.degrees(math.atan2(right_cy - left_cy, right_cx - left_cx + 1e-6))))

    def _gpt_judge(self, features: dict) -> dict:
        user_content = (
            "얼굴 표정 지표를 보고 학생이 현재 수업 내용을 이해하지 못했는지 판단하세요.\n"
            "`confused`는 이해하지 못한 상태 여부입니다.\n"
            "`confidence`는 감정 점수가 아니라 이해하지 못한 정도를 0부터 1 사이 값으로 나타낸 것입니다.\n"
            f"얼굴 표정 지표:\n{json.dumps(features, ensure_ascii=False, indent=2)}"
        )
        confused, gpt_reason, confusion_confidence = False, "", 0.0

        try:
            response = self._gpt.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_completion_tokens=100,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            confused = bool(parsed.get("confused", False))
            confusion_confidence = float(parsed.get("confidence", 0.0))
            confusion_confidence = max(0.0, min(1.0, confusion_confidence))
            gpt_reason = parsed.get("reason", "")
        except json.JSONDecodeError:
            logger.error("GPT response parse failed: %s", raw)
            gpt_reason = "AI 분석 실패"
            confusion_confidence = 0.0
        except (TypeError, ValueError):
            logger.error("GPT confidence parse failed: %s", raw if "raw" in locals() else "")
            gpt_reason = "AI 분석 실패"
            confusion_confidence = 0.0
        except Exception as e:
            logger.error("GPT call error: %s", e)
            gpt_reason = f"GPT 호출 오류: {e}"
            confusion_confidence = 0.0

        logger.info("confused=%s | confidence=%.3f | reason=%s", confused, confusion_confidence, gpt_reason)
        signal_type, signal_subtype, signal_label = self._classify_signal(features, confused)

        return {
            "confused": confused,
            "confidence": confusion_confidence,
            "emotion": features["top_emotion"],
            "gpt_reason": gpt_reason,
            "signal_type": signal_type,
            "signal_subtype": signal_subtype,
            "signal_label": signal_label,
            "face_features": features,
        }

    def summarize(self, audio_text: str) -> dict:
        system_prompt = (
            "당신은 교육 현장에서 강의 내용을 요약하는 도우미입니다.\n"
            "아래 입력은 교육생이 혼란을 느낀 직후 녹음된 강의 전사입니다.\n"
            "반드시 아래 JSON 형식으로만 응답하세요.\n"
            '{"summary": "반드시 2문장 요약", "recommendedConcept": "교육생에게 추가로 설명하면 좋을 개념 한 줄", '
            '"keywords": ["3단어 이하 키워드", "3단어 이하 키워드", "3단어 이하 키워드", "3단어 이하 키워드", "3단어 이하 키워드"]}\n'
            "summary는 반드시 한국어 2문장으로만 작성하세요.\n"
            "recommendedConcept는 교육생이 헷갈렸을 가능성이 높은 개념이나 보충 설명 포인트 한 줄로 작성하세요.\n"
            "keywords는 강의 핵심 키워드를 최대 5개까지 넣고, 각 항목은 반드시 3단어 이하로 작성하세요."
        )
        try:
            response = self._gpt.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"강의 텍스트:\n{audio_text}"},
                ],
                temperature=0.3,
                max_completion_tokens=300,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return {
                "summary": self._normalize_two_sentences(parsed.get("summary", "").strip()),
                "recommendedConcept": parsed.get("recommendedConcept", "").strip(),
                "keywords": self._normalize_keywords(parsed.get("keywords", [])),
            }
        except json.JSONDecodeError:
            logger.error("Summary GPT parse failed: %s", raw if "raw" in locals() else "")
            return {
                "summary": self._normalize_two_sentences(raw if "raw" in locals() else ""),
                "recommendedConcept": "",
                "keywords": [],
            }
        except Exception as e:
            logger.error("Summary GPT call error: %s", e)
            return {
                "summary": f"요약 생성 실패: {e}",
                "recommendedConcept": "",
                "keywords": [],
            }

    def coaching(self, dashboard_data: dict) -> dict:
        system_prompt = (
            "당신은 수업 대시보드를 해석해 강사에게 즉시 실행 가능한 코칭을 제안하는 교육 코치입니다.\n"
            "입력된 데이터만 근거로 판단하고, 과장하지 마세요.\n"
            "반드시 아래 JSON 형식으로만 응답하세요.\n"
            '{"summary":"한두 문장 진단","priorityLevel":"높음|보통|낮음","coachingTips":["팁 1","팁 2","팁 3"],'
            '"reExplainTopics":["주제 1","주제 2","주제 3"],"studentSignals":["신호 해석 1","신호 해석 2"],'
            '"recommendedActionNow":"지금 바로 할 행동 1문장","sampleMentions":["강사가 바로 말할 멘트 1","강사가 바로 말할 멘트 2"]}'
        )
        user_prompt = f"대시보드 데이터:\n{json.dumps(dashboard_data, ensure_ascii=False, indent=2)}"

        try:
            response = self._gpt.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_completion_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)
            return {
                "summary": self._normalize_single_line(parsed.get("summary", "")),
                "priorityLevel": self._normalize_priority(parsed.get("priorityLevel", "보통")),
                "coachingTips": self._normalize_list(parsed.get("coachingTips", []), 3),
                "reExplainTopics": self._normalize_list(parsed.get("reExplainTopics", []), 3),
                "studentSignals": self._normalize_list(parsed.get("studentSignals", []), 3),
                "recommendedActionNow": self._normalize_single_line(parsed.get("recommendedActionNow", "")),
                "sampleMentions": self._normalize_list(parsed.get("sampleMentions", []), 3),
            }
        except json.JSONDecodeError:
            logger.error("Coaching GPT parse failed: %s", raw if "raw" in locals() else "")
        except Exception as e:
            logger.error("Coaching GPT call error: %s", e)

        return self._fallback_coaching(dashboard_data)

    @staticmethod
    def _normalize_two_sentences(text: str) -> str:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return ""

        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        sentences = [part.strip() for part in parts if part.strip()]

        if len(sentences) >= 2:
            return " ".join(sentences[:2])

        if cleaned[-1] not in ".!?":
            cleaned += "."
        return cleaned

    @staticmethod
    def _normalize_keywords(items) -> list[str]:
        if not isinstance(items, list):
            return []

        keywords: list[str] = []
        for item in items:
            if not isinstance(item, str):
                continue

            cleaned = " ".join(item.split()).strip()
            if not cleaned:
                continue

            if len(cleaned.split()) > 3:
                cleaned = " ".join(cleaned.split()[:3])

            if cleaned not in keywords:
                keywords.append(cleaned)

            if len(keywords) >= 5:
                break

        return keywords

    @staticmethod
    def _normalize_single_line(text: str) -> str:
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _normalize_priority(value: str) -> str:
        normalized = " ".join((value or "").split()).strip()
        return normalized if normalized in {"높음", "보통", "낮음"} else "보통"

    @staticmethod
    def _normalize_list(items, limit: int) -> list[str]:
        if not isinstance(items, list):
            return []

        normalized: list[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            cleaned = " ".join(item.split()).strip()
            if not cleaned or cleaned in normalized:
                continue
            normalized.append(cleaned)
            if len(normalized) >= limit:
                break
        return normalized

    def _decode_image(self, image_bytes: bytes):
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _classify_signal(features: dict, confused: bool) -> tuple[str, str, str]:
        if not confused:
            return "FACIAL_INSTABILITY", "LOW_SIGNAL", "표정 기반 불안정"

        if not features.get("face_detected", False):
            return "GAZE_AWAY", "FACE_MISSING", "시선 이탈 / 화면 이탈"

        head_tilt_deg = abs(features.get("head_tilt_deg") or 0.0)
        if head_tilt_deg >= 20:
            return "GAZE_AWAY", "HEAD_TURNED", "시선 이탈 / 화면 이탈"

        ear = features.get("ear")
        if ear is not None and ear > 0 and ear < 0.18:
            return "GAZE_AWAY", "EYES_CLOSED", "시선 이탈 / 화면 이탈"

        top_emotion = str(features.get("top_emotion") or "unknown").strip().lower()
        emotion_subtypes = {
            "fear": "FEAR_DOMINANT",
            "sad": "SAD_DOMINANT",
            "surprise": "SURPRISE_DOMINANT",
            "angry": "ANGRY_TENSION",
            "disgust": "DISGUST_TENSION",
        }
        if top_emotion in emotion_subtypes:
            return "FACIAL_INSTABILITY", emotion_subtypes[top_emotion], "표정 기반 불안정"

        return "FACIAL_INSTABILITY", "CONFUSED_PATTERN", "표정 기반 불안정"

    @staticmethod
    def _fallback(reason: str) -> dict:
        logger.debug("Analysis fallback: %s", reason)
        return {
            "confused": False,
            "confidence": 0.0,
            "emotion": "unknown",
            "gpt_reason": reason,
            "signal_type": "GAZE_AWAY",
            "signal_subtype": "FACE_MISSING",
            "signal_label": "시선 이탈 / 화면 이탈",
            "face_features": {"face_detected": False},
        }

    @staticmethod
    def _fallback_coaching(dashboard_data: dict) -> dict:
        topics = dashboard_data.get("topTopics", []) if isinstance(dashboard_data, dict) else []
        keywords = dashboard_data.get("topKeywords", []) if isinstance(dashboard_data, dict) else []
        signal_items = dashboard_data.get("signalBreakdown", []) if isinstance(dashboard_data, dict) else []
        signals = [item.get("label", "") for item in signal_items if isinstance(item, dict) and item.get("label")]

        focus_topics = [item for item in (topics + keywords) if isinstance(item, str) and item.strip()][:3]
        if not focus_topics:
            focus_topics = ["핵심 개념"]

        return {
            "summary": "현재 대시보드 기준으로 학생 이해도 저하 신호가 반복되고 있어, 핵심 개념을 짧게 다시 정리하는 보충 설명이 필요합니다.",
            "priorityLevel": "보통",
            "coachingTips": [
                "최근 알림이 집중된 개념부터 1분 안에 다시 설명해 주세요.",
                "학생 반응이 큰 반부터 확인 질문을 던져 이해도를 다시 점검해 주세요.",
                "예시 하나와 핵심 정의 하나를 함께 제시해 부담을 낮춰 주세요.",
            ],
            "reExplainTopics": focus_topics,
            "studentSignals": signals[:3] or ["표정과 반응 데이터를 함께 보며 보충 설명 우선순위를 잡아 주세요."],
            "recommendedActionNow": f"지금은 '{focus_topics[0]}'부터 짧게 다시 설명하고 바로 이해 확인 질문을 해보세요.",
            "sampleMentions": [
                f"방금 나온 '{focus_topics[0]}'를 다시 아주 짧게 정리해볼게요.",
                "지금 설명한 내용 중 어디까지 이해됐는지 한 번 같이 확인해볼게요.",
            ],
        }
