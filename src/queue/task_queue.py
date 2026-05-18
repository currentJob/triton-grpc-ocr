"""
Async Task Queue — 비동기 작업 제출 및 결과 조회.

패턴: Producer-Consumer + Observer
  - submit()   : 작업을 큐에 넣고 task_id 반환 (논블로킹)
  - get_result(): task_id로 결과 폴링
  - start_worker(): 백그라운드에서 큐를 소비

학습 포인트:
  - asyncio.Queue로 스레드 없이 동시성 처리
  - 상태 머신: PENDING → RUNNING → DONE | FAILED
  - 실제 서비스라면 Redis + Celery 또는 RQ로 대체
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


@dataclass
class Task:
    id:         str
    image_path: str
    created_at: float = field(default_factory=time.time)
    metadata:   dict  = field(default_factory=dict)


@dataclass
class TaskResult:
    task_id:    str
    status:     TaskStatus
    created_at: float
    started_at: float  = 0.0
    finished_at: float = 0.0
    data:       Any    = None
    error:      str    = ""

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at if self.finished_at else 0.0


# 처리 함수 타입: (Task) -> 결과 dict
ProcessFn = Callable[[Task], Awaitable[dict]]


class TaskQueue:
    def __init__(self, max_size: int = 100):
        self._queue:   asyncio.Queue[Task]     = asyncio.Queue(maxsize=max_size)
        self._results: dict[str, TaskResult]   = {}
        self._running: bool                    = False

    # ── 외부 API ──────────────────────────────────────────────────────────────

    async def submit(self, image_path: str, metadata: dict | None = None) -> str:
        task = Task(id=str(uuid.uuid4()), image_path=image_path, metadata=metadata or {})
        self._results[task.id] = TaskResult(
            task_id=task.id, status=TaskStatus.PENDING, created_at=task.created_at
        )
        await self._queue.put(task)
        return task.id

    def get_result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    def list_results(self, limit: int = 20) -> list[TaskResult]:
        return sorted(self._results.values(), key=lambda r: r.created_at, reverse=True)[:limit]

    def queue_size(self) -> int:
        return self._queue.qsize()

    # ── Worker ────────────────────────────────────────────────────────────────

    async def start_worker(self, process_fn: ProcessFn) -> None:
        """백그라운드 워커. FastAPI lifespan 안에서 asyncio.create_task()로 실행."""
        self._running = True
        while self._running:
            try:
                task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            result = self._results[task.id]
            result.status     = TaskStatus.RUNNING
            result.started_at = time.time()

            try:
                result.data   = await process_fn(task)
                result.status = TaskStatus.DONE
            except Exception as e:
                result.error  = str(e)
                result.status = TaskStatus.FAILED
            finally:
                result.finished_at = time.time()
                self._queue.task_done()

    def stop(self) -> None:
        self._running = False
