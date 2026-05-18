"""
Pipeline Context — 각 Stage가 읽고 쓰는 공유 데이터 컨테이너.

패턴: Chain-of-Responsibility + Context Object
  - Stage마다 Context를 변경하고 다음 Stage로 전달
  - 어느 Stage에서든 `error`를 설정하면 이후 Stage는 실행되지 않음
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class OCRItem:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2


@dataclass
class PipelineContext:
    # ── 입력 ──────────────────────────────────────────────────────
    image_path: str
    task_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── Stage 결과 ────────────────────────────────────────────────
    image_size: tuple[int, int] = (0, 0)        # (width, height)
    raw_bboxes: list[tuple[int,int,int,int]] = field(default_factory=list)
    ocr_items: list[OCRItem] = field(default_factory=list)

    # ── 제어 / 관측 ───────────────────────────────────────────────
    error: str = ""
    stage_timings: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    @property
    def failed(self) -> bool:
        return bool(self.error)

    @property
    def texts(self) -> list[str]:
        return [item.text for item in self.ocr_items if item.text]

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at
