import asyncio
import os
import tempfile
from contextlib import asynccontextmanager

import grpc
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.generated import inference_pb2, inference_pb2_grpc
from src.observability import setup_logging
from src.observability.metrics import get_metrics
from src.ocr_pipeline import OCRPipeline
from src.queue import TaskQueue
from src.store import OCRRecord, ResultStore, VectorStore

CHAR_DICT    = "model_repository/_config/charDict.json"
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "localhost")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "50055"))

setup_logging(level=os.getenv("LOG_LEVEL", "INFO"), json_format=False)

# ── 전역 싱글턴 ───────────────────────────────────────────────────────────────
_pipeline:     OCRPipeline  | None = None
_task_queue:   TaskQueue    | None = None
_result_store: ResultStore  | None = None
_vector_store: VectorStore  | None = None
_worker_task:  asyncio.Task | None = None


# ── 비동기 작업 처리 함수 ─────────────────────────────────────────────────────
async def _process_task(task) -> dict:
    """TaskQueue 워커가 호출하는 실제 처리 함수."""
    metrics = get_metrics()
    with metrics.measure("pipeline.total"):
        results = _pipeline.run(task.image_path)

    texts   = [r.text for r in results if r.text]
    elapsed = sum(metrics.summary()["timings"].get("pipeline.total", {}).get("mean", 0) for _ in [1])

    record = ResultStore.make_record(task.id, task.image_path, texts, elapsed)
    await _result_store.save(record)
    _vector_store.add(" ".join(texts), {"record_id": record.id, "task_id": task.id})

    metrics.increment("task.done")
    return {"record_id": record.id, "count": len(results)}


# ── 앱 생명주기 ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline, _task_queue, _result_store, _vector_store, _worker_task

    _pipeline     = OCRPipeline(char_dict_path=CHAR_DICT,
                                gateway_host=GATEWAY_HOST, gateway_port=GATEWAY_PORT)
    _result_store = ResultStore()
    _vector_store = VectorStore()
    _task_queue   = TaskQueue()

    await _result_store.init()
    _worker_task = asyncio.create_task(_task_queue.start_worker(_process_task))

    yield

    _task_queue.stop()
    _worker_task.cancel()


app = FastAPI(title="OCR API", version="2.0", lifespan=lifespan)

# ── 헬스 체크 ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    MB = 1024 * 1024
    opts = [("grpc.max_receive_message_length", 256 * MB)]
    try:
        with grpc.insecure_channel(f"{GATEWAY_HOST}:{GATEWAY_PORT}", options=opts) as ch:
            stub = inference_pb2_grpc.InferenceServiceStub(ch)
            h    = stub.ServerHealth(inference_pb2.HealthRequest())
            models = ["ocr_det", "ocr_rec", "doc_ori", "textline_ori"]
            return {
                "live":   h.live,
                "ready":  h.ready,
                "models": {m: stub.ModelReady(inference_pb2.ModelReadyRequest(model_name=m)).ready for m in models},
                "queue":  _task_queue.queue_size() if _task_queue else 0,
            }
    except Exception as e:
        return {"live": False, "ready": False, "models": {}, "error": str(e)}

# ── OCR 동기 (기존) ──────────────────────────────────────────────────────────
@app.post("/api/ocr")
async def ocr(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        img_w, img_h = Image.open(tmp_path).size
        metrics = get_metrics()
        with metrics.measure("api.ocr.sync"):
            results = _pipeline.run(tmp_path)
        metrics.increment("api.ocr.sync.requests")
        return {
            "image_size": {"width": img_w, "height": img_h},
            "count": len(results),
            "results": [
                {"index": i+1, "text": r.text, "confidence": round(r.confidence, 4),
                 "bbox": {"x1": r.bbox[0], "y1": r.bbox[1], "x2": r.bbox[2], "y2": r.bbox[3]}}
                for i, r in enumerate(results)
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

# ── OCR 비동기 (Task Queue) ──────────────────────────────────────────────────
@app.post("/api/ocr/async", status_code=202)
async def ocr_async(file: UploadFile = File(...)):
    """이미지를 큐에 제출하고 즉시 task_id 반환. 결과는 /api/task/{id}로 폴링."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")

    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(await file.read())
    tmp.close()

    task_id = await _task_queue.submit(tmp.name)
    get_metrics().increment("api.ocr.async.submitted")
    return {"task_id": task_id, "status": "pending"}

@app.get("/api/task/{task_id}")
def get_task(task_id: str):
    result = _task_queue.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="task_id를 찾을 수 없습니다.")
    return {
        "task_id":   result.task_id,
        "status":    result.status,
        "duration":  round(result.duration, 3),
        "data":      result.data,
        "error":     result.error,
    }

@app.get("/api/tasks")
def list_tasks(limit: int = 20):
    return [
        {"task_id": r.task_id, "status": r.status, "duration": round(r.duration, 3)}
        for r in _task_queue.list_results(limit)
    ]

# ── 히스토리 (SQLite) ────────────────────────────────────────────────────────
@app.get("/api/history")
async def history(limit: int = 20):
    records = await _result_store.list_recent(limit)
    return [
        {"id": r.id, "texts": r.texts, "item_count": r.item_count,
         "elapsed": round(r.elapsed, 3), "created_at": r.created_at}
        for r in records
    ]

@app.delete("/api/history/{record_id}")
async def delete_record(record_id: str):
    ok = await _result_store.delete(record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="레코드를 찾을 수 없습니다.")
    return {"deleted": record_id}

# ── RAG 검색 ─────────────────────────────────────────────────────────────────
@app.get("/api/rag/search")
def rag_search(q: str, k: int = 5):
    """
    OCR 히스토리에서 쿼리와 유사한 문서 검색.
    결과를 LLM 프롬프트에 주입하면 RAG 완성.
    """
    results = _vector_store.search(q, k=k)
    return {
        "query":   q,
        "count":   len(results),
        "indexed": _vector_store.size(),
        "results": [{"id": r.id, "text": r.text, "score": r.score} for r in results],
    }

# ── 메트릭스 ──────────────────────────────────────────────────────────────────
@app.get("/api/metrics")
def metrics():
    return get_metrics().summary()

# ── 기타 ──────────────────────────────────────────────────────────────────────
@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def devtools_json():
    return Response(status_code=204)

app.mount("/", StaticFiles(directory="src/static", html=True), name="static")


def serve(host: str = "0.0.0.0", port: int = 8080):
    print(f"[WebServer] http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
