# Confusion Detection API
> 교육생 혼란 감지 FastAPI 서비스

---

## 1. 프로젝트 개요

교육 현장에서 교육생 웹캠 영상을 10초마다 캡처해 얼굴 표정을 분석하고, 혼란(Confusion) 여부를 실시간으로 판정하는 FastAPI 기반 마이크로서비스입니다.

FER과 MediaPipe로 원시 표정 데이터를 추출한 뒤, OpenAI GPT 모델이 각 지표의 의미를 종합하여 이해도를 최종 판정합니다.

| 항목 | 내용 |
|------|------|
| 언어 | Python 3.11 |
| 프레임워크 | FastAPI 0.111 / Uvicorn 0.29 |
| 표정 분석 | FER 22.5.1 + TensorFlow 2.20 |
| 랜드마크 | MediaPipe 0.10.33 (Face Landmarker Tasks API) |
| AI 판정 | OpenAI GPT — 기본 모델 `gpt-5.4-mini` (.env 에서 변경 가능) |
| 컨테이너 | Docker / Docker Compose |

---

## 2. 전체 시스템 아키텍처

이 서비스는 전체 시스템의 분석 레이어입니다.

```
[교육생 브라우저]
  웹캠 → 10초마다 프레임 캡처
  → REST POST /analyze/{studentId}   ← 이 서비스로 전송
  ← { confused: true/false } 수신
  → 연속 3회(30초) confused=true 이면 Spring으로 이벤트 전송

[FastAPI]  ← 이 서비스
  이미지 수신
  → FER      : 7가지 감정 확률 추출
  → MediaPipe: 눈썹비율 / 눈개폐율 / 고개기울기 추출
  → GPT      : 원시 데이터 전달 → confused true/false 판정
  ← 응답 반환

[Spring]
  교육생 이벤트 수신 { studentId, capturedAt }
  → 최근 30초 기준 과반수 판정
  → 충족 시 Alert DB 저장
  → WebSocket으로 강사 알림 전송

[강사 브라우저]
  WebSocket 알림 수신 → 대시보드 표시
  → 반별 알림 이력 / 어려워한 시점 타임라인 시각화
```

---

## 3. 파일 구조

```
fastapi/
├── main.py                  # FastAPI 앱 진입점, 엔드포인트 정의
├── analyzer.py              # 핵심 분석 로직 (FER + MediaPipe + GPT)
├── models.py                # Pydantic 요청/응답 스키마
├── requirements.txt         # Python 의존성 목록
├── Dockerfile               # 컨테이너 이미지 빌드 설정
├── docker-compose.yml       # 컨테이너 실행 설정
├── .env                     # 환경변수 (git 제외)
├── .env.example             # 환경변수 템플릿
├── .gitignore               # face_landmarker.task, .env 등 제외
└── face_landmarker.task     # MediaPipe 모델 (첫 실행 시 자동 다운로드, git 제외)
```

---

## 4. 분석 흐름 상세 (analyzer.py)

이미지 1장이 POST 요청으로 들어오면 아래 3단계를 순서대로 처리합니다.

### STEP 1 — FER: 감정 확률 추출

FER(Facial Expression Recognition)은 딥러닝 기반 표정 인식 라이브러리입니다. 정적 이미지 한 장에서 7가지 감정을 분류하며, 각 감정 점수의 합은 1.0입니다.

| 감정 | 의미 |
|------|------|
| `happy` | 밝고 편안한 표정 |
| `neutral` | 무표정. 집중 또는 수동적으로 듣는 상태일 수 있음 |
| `surprise` | 놀람. 예상치 못한 내용에 반응하는 것으로 혼란과 구분 필요 |
| `fear` | 두려움 또는 당혹감. 내용이 어렵거나 압도될 때 나타날 수 있음 |
| `sad` | 슬픔 또는 무기력감. 이해를 포기하거나 좌절하는 상태와 연관 |
| `angry` | 짜증 또는 답답함. 이해가 안 될 때 발생하는 내적 저항 표현 |
| `disgust` | 불쾌감. 내용이 맞지 않거나 거부감을 느낄 때 나타날 수 있음 |

> **신뢰도 한계**: 조명·각도·해상도에 따라 정확도가 달라지며, `neutral`이 높다고 해서 반드시 이해한 상태는 아닙니다 (집중 또는 무관심일 수도 있음).

### STEP 2 — MediaPipe: 얼굴 랜드마크 지표 추출

Google MediaPipe Face Landmarker로 478개 얼굴 특징점 좌표를 추출하여 3가지 기하학적 수치를 계산합니다. 감정을 직접 감지하는 것이 아니라 얼굴의 형태적 변화를 수치화합니다.

| 지표 | 의미 | 혼란 신호 해석 |
|------|------|---------------|
| `brow_eye_ratio` | 눈썹과 눈 사이의 수직 거리 비율 | 낮을수록 눈썹이 찡그려진 상태 |
| `ear` | Eye Aspect Ratio (눈 세로/가로 비율) | 낮으면 졸음, 높으면 긴장 또는 놀람 |
| `head_tilt_deg` | 고개 기울기 각도 (도 단위) | 클수록 의아하거나 생각 중인 상태 |
| `face_detected` | 얼굴 감지 여부 (true/false) | false이면 자리 이탈 또는 카메라 이탈 |

> **신뢰도 한계**: 정면 얼굴 기준으로 설계되어 고개가 많이 돌아가거나 조명이 어두우면 정확도가 낮아질 수 있습니다.

### STEP 3 — GPT: 이해도 최종 판정

STEP 1~2에서 추출한 모든 원시 데이터를 GPT에 전달하여 혼란 여부를 종합 판단합니다. 임계값을 코드에 하드코딩하지 않고, GPT가 각 지표의 의미와 도구의 한계를 이해한 뒤 스스로 판단합니다.

- 시스템 프롬프트: FER/MediaPipe 도구 설명, 각 지표 의미, 신뢰도 한계 포함
- GPT 응답 형식: `{ "confused": true/false, "reason": "판단 이유 한 문장" }`
- 기본 모델: `gpt-5.4-mini` — `.env`의 `OPENAI_MODEL`로 변경 가능

---

## 5. API 명세

### POST `/analyze/{student_id}`

교육생 웹캠 캡처 이미지를 분석하여 혼란 여부를 반환합니다.

| 항목 | 내용 |
|------|------|
| Method | POST |
| Content-Type | multipart/form-data |
| Path Param | `student_id` — 교육생 식별자 (Spring과 동일 ID 사용) |
| Body | `file` — 이미지 파일 (JPEG / PNG / WebP, 최대 10MB) |

**응답 예시 (200 OK)**

```json
{
  "studentId":   "student_42",
  "confused":    true,
  "confidence":  0.712,
  "emotion":     "fear",
  "gpt_reason":  "fear 수치가 높고 눈썹이 찡그려진 상태로 혼란 신호가 명확합니다.",
  "face_features": {
    "face_detected":  true,
    "emotions":       { "happy": 0.02, "neutral": 0.20, "fear": 0.41, "sad": 0.12, "angry": 0.10, "disgust": 0.08, "surprise": 0.07 },
    "top_emotion":    "fear",
    "confidence":     0.41,
    "brow_eye_ratio": 0.038,
    "ear":            0.261,
    "head_tilt_deg":  3.1
  }
}
```

### GET `/health`

서버 상태 및 분석기 초기화 여부를 확인합니다.

```json
{ "status": "ok", "analyzer_ready": true }
```

---

## 6. 환경변수

`.env.example`을 복사해 `.env`로 저장한 뒤 값을 입력합니다.

```bash
cp .env.example .env
```

| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 (필수) — `sk-...` |
| `OPENAI_MODEL` | GPT 모델명 (기본값: `gpt-5.4-mini`) |

---

## 7. 설치 및 실행

### 로컬 실행

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 3. 서버 실행
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

> **Windows 한글 경로 주의**: MediaPipe C++ 라이브러리가 한글 경로를 읽지 못합니다.
> `face_landmarker.task` 모델 파일은 홈 디렉터리(`C:\Users\계정명\`)에 자동 저장됩니다.
> 배포 환경(Linux)에서는 코드 폴더에 저장되며 문제가 없습니다.

### Docker 실행

```bash
# 빌드 및 백그라운드 실행
docker-compose up --build -d

# 로그 확인
docker-compose logs -f

# 중지 및 컨테이너 제거
docker-compose down
```

---

## 8. Jenkins 배포

`docker-compose down`을 먼저 실행해 이전 컨테이너를 완전히 제거해야 `ContainerConfig` 오류를 방지할 수 있습니다.

```groovy
sh "docker-compose -f docker-compose.yml down || true"
sh "docker-compose -f docker-compose.yml up -d --build"
```

**사전 조건**
- Jenkins 유저가 docker 그룹에 속해야 합니다: `sudo usermod -aG docker jenkins`
- `.env` 파일이 Jenkins workspace에 복사되어 있어야 합니다
- 서버 첫 배포 시 `face_landmarker.task` (약 5MB) 가 자동 다운로드됩니다

---

## 9. 주요 의존성

| 패키지 | 버전 | 용도 |
|--------|------|------|
| `fastapi` | 0.111.0 | 웹 프레임워크 |
| `uvicorn[standard]` | 0.29.0 | ASGI 서버 |
| `python-multipart` | 0.0.9 | multipart/form-data 파싱 |
| `fer` | 22.5.1 | 얼굴 감정 인식 (7가지 감정) |
| `tensorflow` | 2.20.0 | FER 딥러닝 백엔드 |
| `mediapipe` | 0.10.33 | 얼굴 랜드마크 추출 (478점) |
| `opencv-python-headless` | 최신 | 이미지 디코딩 및 처리 |
| `numpy` | >=2.0.0 | 수치 연산 |
| `openai` | >=1.30.0 | GPT API 호출 |
| `python-dotenv` | >=1.0.0 | 환경변수 로드 |
| `moviepy` | 최신 | FER 내부 의존성 |
