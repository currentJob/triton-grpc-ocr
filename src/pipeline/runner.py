"""
Pipeline Runner — Stage 목록을 순서대로 실행.

패턴: Composite + Chain of Responsibility
  - 어느 Stage에서 error가 발생하면 이후 Stage를 건너뜀
  - 각 Stage의 실행 시간이 Context에 자동 기록됨
"""
from __future__ import annotations

import logging

from .context import PipelineContext
from .stages import Stage

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, stages: list[Stage]):
        self._stages = stages

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        logger.info("pipeline.start task_id=%s image=%s", ctx.task_id, ctx.image_path)

        for stage in self._stages:
            name = type(stage).__name__
            if ctx.failed:
                logger.warning("pipeline.skip stage=%s reason=prior_error", name)
                continue
            logger.debug("pipeline.stage stage=%s", name)
            ctx = await stage.run(ctx)

        if ctx.failed:
            logger.error("pipeline.failed task_id=%s error=%s", ctx.task_id, ctx.error)
        else:
            logger.info(
                "pipeline.done task_id=%s texts=%d elapsed=%.3fs timings=%s",
                ctx.task_id, len(ctx.ocr_items), ctx.elapsed, ctx.stage_timings,
            )
        return ctx
