"""
Result Store — OCR 결과를 SQLite에 영구 저장.

학습 포인트:
  - aiosqlite로 비동기 DB I/O (블로킹 없이)
  - Repository 패턴: DB 세부사항을 비즈니스 로직과 분리
  - 실제 서비스라면 PostgreSQL + SQLAlchemy async로 대체
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import aiosqlite


@dataclass
class OCRRecord:
    id:         str
    task_id:    str
    image_path: str
    texts:      list[str]           # 인식된 텍스트 목록
    item_count: int
    elapsed:    float               # 처리 시간 (초)
    created_at: float

    @property
    def full_text(self) -> str:
        return " ".join(self.texts)


class ResultStore:
    def __init__(self, db_path: str = "data/ocr_results.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init(self) -> None:
        """앱 시작 시 한 번 호출."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS ocr_results (
                    id          TEXT PRIMARY KEY,
                    task_id     TEXT NOT NULL,
                    image_path  TEXT NOT NULL,
                    texts       TEXT NOT NULL,   -- JSON array
                    item_count  INTEGER,
                    elapsed     REAL,
                    created_at  REAL NOT NULL
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON ocr_results(created_at DESC)")
            await db.commit()

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def save(self, record: OCRRecord) -> str:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO ocr_results VALUES (?,?,?,?,?,?,?)",
                (record.id, record.task_id, record.image_path,
                 json.dumps(record.texts, ensure_ascii=False),
                 record.item_count, record.elapsed, record.created_at),
            )
            await db.commit()
        return record.id

    async def get(self, record_id: str) -> OCRRecord | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM ocr_results WHERE id = ?", (record_id,)
            ) as cur:
                row = await cur.fetchone()
        return self._row_to_record(row) if row else None

    async def list_recent(self, limit: int = 20) -> list[OCRRecord]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT * FROM ocr_results ORDER BY created_at DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_record(r) for r in rows]

    async def delete(self, record_id: str) -> bool:
        async with aiosqlite.connect(self._db_path) as db:
            cur = await db.execute("DELETE FROM ocr_results WHERE id = ?", (record_id,))
            await db.commit()
            return cur.rowcount > 0

    # ── 팩토리 ────────────────────────────────────────────────────────────────

    @staticmethod
    def make_record(task_id: str, image_path: str, texts: list[str],
                    elapsed: float) -> OCRRecord:
        return OCRRecord(
            id=str(uuid.uuid4()), task_id=task_id, image_path=image_path,
            texts=texts, item_count=len(texts), elapsed=elapsed,
            created_at=time.time(),
        )

    @staticmethod
    def _row_to_record(row: tuple) -> OCRRecord:
        id_, task_id, image_path, texts_json, item_count, elapsed, created_at = row
        return OCRRecord(
            id=id_, task_id=task_id, image_path=image_path,
            texts=json.loads(texts_json), item_count=item_count,
            elapsed=elapsed, created_at=created_at,
        )
