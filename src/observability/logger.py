"""
구조적 로깅 설정 — JSON 포맷으로 파싱하기 쉬운 로그 출력.

학습 포인트:
  - 구조적 로그(JSON)는 ELK/Loki 같은 로그 집계 시스템에 바로 연동 가능
  - 실제 서비스라면 OpenTelemetry로 트레이싱까지 확장
"""
from __future__ import annotations

import json
import logging
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log: dict = {
            "ts":      round(time.time(), 3),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        return json.dumps(log, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if json_format else logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s  %(message)s"
    ))
    logging.basicConfig(level=getattr(logging, level.upper()), handlers=[handler], force=True)
    # 외부 라이브러리 로그 억제
    for noisy in ("uvicorn.access", "httpx", "hpack"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
