"""
Metrics — 처리 시간·카운터·에러율 추적.

학습 포인트:
  - Context Manager로 자동 타이밍 측정
  - 인메모리 집계 (평균, P50, P95, P99)
  - 실제 서비스라면 Prometheus + Grafana로 대체
"""
from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class _Stat:
    values: list[float] = field(default_factory=list)
    count:  int         = 0
    total:  float       = 0.0
    errors: int         = 0

    def record(self, v: float) -> None:
        self.values.append(v)
        self.count += 1
        self.total += v

    def percentile(self, p: float) -> float:
        if not self.values:
            return 0.0
        s = sorted(self.values)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    def summary(self) -> dict:
        return {
            "count":   self.count,
            "mean":    round(self.total / self.count, 4) if self.count else 0,
            "p50":     round(self.percentile(50), 4),
            "p95":     round(self.percentile(95), 4),
            "p99":     round(self.percentile(99), 4),
            "errors":  self.errors,
        }


class Metrics:
    def __init__(self):
        self._stats:    defaultdict[str, _Stat] = defaultdict(_Stat)
        self._counters: defaultdict[str, int]   = defaultdict(int)

    def record(self, name: str, value: float) -> None:
        self._stats[name].record(value)

    def increment(self, name: str, n: int = 1) -> None:
        self._counters[name] += n

    def error(self, name: str) -> None:
        self._stats[name].errors += 1
        self._counters[f"{name}.errors"] += 1

    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """with metrics.measure("ocr.detect"): ... 으로 자동 타이밍."""
        start = time.perf_counter()
        ok = True
        try:
            yield
        except Exception:
            ok = False
            self.error(name)
            raise
        finally:
            elapsed = time.perf_counter() - start
            self._stats[name].record(elapsed)
            if ok:
                self.increment(f"{name}.ok")

    def summary(self) -> dict:
        return {
            "timings":  {k: v.summary() for k, v in self._stats.items()},
            "counters": dict(self._counters),
        }


# 모듈 수준 싱글턴
_global = Metrics()


@contextmanager
def timer(name: str) -> Generator[None, None, None]:
    with _global.measure(name):
        yield


def get_metrics() -> Metrics:
    return _global
