# grpc-demo-page

PaddleOCR 모델을 NVIDIA Triton Inference Server로 서빙하고, gRPC 게이트웨이를 통해 추론 요청을 처리하는 데모 프로젝트입니다.  
React 프론트엔드에서 이미지를 업로드하거나 카메라로 촬영하면 OCR 결과와 함께 LLM 후처리까지 확인할 수 있습니다.

---

## 아키텍처

```
[Browser / Client]
       │  HTTP / REST
       ▼
[Web Server] FastAPI  :8080
       │  gRPC (InferenceService)
       ▼
[Inference Gateway]   :50055
       │  tritonclient[grpc]
       ▼
[Triton Inference Server]  :8001 (gRPC) | :8000 (HTTP) | :8002 (Metrics)
       │
  ┌────┴────┐
ocr_det   ocr_rec   doc_ori   textline_ori
```

| 레이어 | 역할 |
|--------|------|
| Triton | 모델 로딩 및 GPU 추론 |
| Inference Gateway | Triton 앞단 gRPC 게이트웨이. 커스텀 proto 스키마로 클라이언트를 Triton 내부 구조에서 분리 |
| Web Server | REST API + 정적 프론트엔드 서빙. 비동기 Task Queue, SQLite 히스토리, Vector Store(RAG) 내장 |

---

## 주요 기능

- **OCR 파이프라인** — PaddleOCR 기반 2단계 처리
  - `ocr_det`: DB 알고리즘으로 텍스트 영역 검출
  - `ocr_rec`: CTC 디코딩으로 텍스트 인식
- **동기 / 비동기 OCR API** — 즉시 응답 또는 Task Queue를 통한 폴링
- **히스토리** — SQLite에 OCR 결과 저장 및 조회
- **RAG 검색** — 벡터 스토어로 OCR 히스토리에서 유사 문서 검색
- **React 프론트엔드** — 이미지 업로드, 카메라 캡처, ROI 선택, LLM 연동
- **Observability** — 구조화 로깅, 내부 메트릭스 엔드포인트

---

## 요구사항

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- Docker & Docker Compose
- NVIDIA GPU (선택 — CPU 전용 실행도 가능)

---

## 빠른 시작

### 1. 의존성 설치

```bash
uv pip install -e .
```

### 2. Proto 코드 생성

```bash
scripts\generate_proto.bat
```

### 3. Docker로 Triton 실행

```bash
docker compose up triton
```

GPU를 사용하려면 `docker-compose.yml`의 `deploy` 섹션 주석을 해제하세요.

### 4. 서버 실행

**Inference Gateway (gRPC)**

```bash
python main.py --service inference --triton-host localhost --triton-port 8001
```

**Web Server (REST + 프론트엔드)**

```bash
python main.py --service web
```

브라우저에서 `http://localhost:8080` 접속

---

## Docker Compose로 전체 실행

```bash
docker compose up
```

| 서비스 | 포트 |
|--------|------|
| Triton HTTP | 8000 |
| Triton gRPC | 8001 |
| Triton Metrics | 8002 |
| Inference Gateway | 50055 |

---

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET` | `/api/health` | 서버 및 모델 상태 확인 |
| `POST` | `/api/ocr` | 동기 OCR (즉시 결과 반환) |
| `POST` | `/api/ocr/async` | 비동기 OCR (task_id 반환) |
| `GET` | `/api/task/{task_id}` | 비동기 작업 결과 폴링 |
| `GET` | `/api/tasks` | 전체 작업 목록 |
| `GET` | `/api/history` | OCR 히스토리 조회 |
| `DELETE` | `/api/history/{id}` | 히스토리 레코드 삭제 |
| `GET` | `/api/rag/search?q=...` | 벡터 검색 (RAG) |
| `GET` | `/api/metrics` | 내부 메트릭스 |

### 동기 OCR 예시

```bash
curl -X POST http://localhost:8080/api/ocr \
  -F "file=@sample.jpg"
```

```json
{
  "image_size": { "width": 1920, "height": 1080 },
  "count": 3,
  "results": [
    { "index": 1, "text": "Hello", "confidence": 0.9821, "bbox": { "x1": 10, "y1": 20, "x2": 100, "y2": 50 } }
  ]
}
```

---

## gRPC 서비스 정의

```proto
service InferenceService {
  rpc Infer        (InferRequest)          returns (InferResponse);
  rpc InferStream  (stream InferRequest)   returns (stream InferResponse);
  rpc ServerHealth (HealthRequest)         returns (HealthResponse);
  rpc ModelReady   (ModelReadyRequest)     returns (ModelReadyResponse);
}
```

---

## 프로젝트 구조

```
.
├── main.py                  # 진입점 (--service inference | web)
├── proto/
│   └── inference.proto      # gRPC 서비스 정의
├── src/
│   ├── inference_server.py  # gRPC 게이트웨이 서버
│   ├── inference_servicer.py
│   ├── triton_client.py     # Triton gRPC 클라이언트 래퍼
│   ├── web_server.py        # FastAPI 애플리케이션
│   ├── ocr_pipeline.py      # OCR 파이프라인 (단순 버전)
│   ├── generated/           # proto 빌드 결과물
│   ├── pipeline/            # 스테이지 기반 파이프라인 (Strategy 패턴)
│   ├── queue/               # 비동기 Task Queue
│   ├── store/               # ResultStore(SQLite) + VectorStore
│   ├── observability/       # 로깅 + 메트릭스
│   └── static/              # 빌드된 React 정적 파일
├── frontend/                # React + TypeScript 소스
├── model_repository/        # Triton 모델 저장소
├── scripts/
│   ├── generate_proto.bat   # proto 코드 생성 스크립트
│   └── test_image.py
├── Dockerfile
└── docker-compose.yml
```

---

## 개발

### Proto 재생성

`proto/inference.proto` 수정 후:

```bash
scripts\generate_proto.bat
```

### 프론트엔드 빌드

```bash
cd frontend
npm install
npm run build
```

빌드 결과물은 `src/static/`에 복사하세요.

---

## CLI 옵션

```
python main.py [--service {inference,web}]
               [--triton-host HOST]
               [--triton-port PORT]
               [--port PORT]
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--service` | `inference` | 실행할 서비스 선택 |
| `--triton-host` | `localhost` | Triton 서버 호스트 |
| `--triton-port` | `8001` | Triton gRPC 포트 |
| `--port` | inference=50055, web=8080 | 게이트웨이/웹 서버 포트 |
