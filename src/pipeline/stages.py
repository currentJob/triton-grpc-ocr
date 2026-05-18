"""
Pipeline Stages — 각 처리 단계를 독립 클래스로 분리.

패턴: Strategy + Template Method
  - 모든 Stage는 동일한 인터페이스(run)를 구현
  - 새 Stage 추가 시 기존 코드 변경 없음 (Open/Closed Principle)
"""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np
from PIL import Image

from .context import OCRItem, PipelineContext


# ── 인터페이스 ────────────────────────────────────────────────────────────────

class Stage(Protocol):
    """모든 Pipeline Stage가 구현해야 하는 인터페이스."""
    async def run(self, ctx: PipelineContext) -> PipelineContext: ...


# ── 구체 Stage ────────────────────────────────────────────────────────────────

class DecodeStage:
    """이미지 파일을 로드하고 기본 정보를 추출."""

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        t = time.perf_counter()
        try:
            img = Image.open(ctx.image_path).convert("RGB")
            ctx.image_size = img.size           # (width, height)
            ctx.metadata["format"] = img.format or "UNKNOWN"
            ctx.metadata["mode"]   = img.mode
        except Exception as e:
            ctx.error = f"[DecodeStage] {e}"
        finally:
            ctx.stage_timings["decode"] = time.perf_counter() - t
        return ctx


class DetectStage:
    """OCR 검출 모델로 텍스트 영역 bbox를 추출."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, triton_infer_fn, thresh: float = 0.3, max_side: int = 960):
        """
        triton_infer_fn: (model, input_name, data, output_name) -> np.ndarray
        """
        self._infer  = triton_infer_fn
        self._thresh = thresh
        self._max_side = max_side

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.failed:
            return ctx
        t = time.perf_counter()
        try:
            img = Image.open(ctx.image_path).convert("RGB")
            orig_w, orig_h = img.size
            tensor, scale_h, scale_w = self._preprocess(img)

            prob_map = self._infer("ocr_det", "x", tensor, "fetch_name_0")[0, 0]
            binary   = (prob_map > self._thresh).astype(np.uint8)
            bboxes   = self._extract_bboxes(binary)

            ctx.raw_bboxes = [
                (
                    max(0, int(x1 / scale_w)),
                    max(0, int(y1 / scale_h)),
                    min(orig_w, int(x2 / scale_w)),
                    min(orig_h, int(y2 / scale_h)),
                )
                for x1, y1, x2, y2 in bboxes
            ]
        except Exception as e:
            ctx.error = f"[DetectStage] {e}"
        finally:
            ctx.stage_timings["detect"] = time.perf_counter() - t
        return ctx

    def _preprocess(self, img: Image.Image) -> tuple[np.ndarray, float, float]:
        orig_w, orig_h = img.size
        scale = min(self._max_side / max(orig_h, orig_w), 1.0)
        new_h = max(int(orig_h * scale / 32) * 32, 32)
        new_w = max(int(orig_w * scale / 32) * 32, 32)
        arr = np.array(img.resize((new_w, new_h), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = ((arr - self.MEAN) / self.STD).transpose(2, 0, 1)[np.newaxis]
        return arr, new_h / orig_h, new_w / orig_w

    @staticmethod
    def _extract_bboxes(binary: np.ndarray, min_area: int = 100) -> list[tuple]:
        h, w = binary.shape
        dilated = np.zeros_like(binary)
        for r in range(h):
            for c in np.where(binary[r])[0]:
                dilated[r, max(0, c - 8):min(w, c + 9)] = 1

        visited = np.zeros_like(dilated, dtype=bool)
        bboxes  = []
        for r in range(h):
            for c in range(w):
                if not dilated[r, c] or visited[r, c]:
                    continue
                queue = [(r, c)]
                visited[r, c] = True
                min_r = max_r = r
                min_c = max_c = c
                while queue:
                    cr, cc = queue.pop()
                    min_r, max_r = min(min_r, cr), max(max_r, cr)
                    min_c, max_c = min(min_c, cc), max(max_c, cc)
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<h and 0<=nc<w and dilated[nr,nc] and not visited[nr,nc]:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
                if (max_r-min_r+1)*(max_c-min_c+1) >= min_area:
                    bboxes.append((min_c, min_r, max_c, max_r))
        return bboxes


class RecognizeStage:
    """각 bbox 영역을 인식 모델에 넣어 텍스트 추출."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, triton_infer_fn, char_table: list, blank_idx: int, rec_height: int = 48):
        self._infer      = triton_infer_fn
        self._char_table = char_table
        self._blank_idx  = blank_idx
        self._rec_height = rec_height

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.failed or not ctx.raw_bboxes:
            return ctx
        t = time.perf_counter()
        try:
            img = Image.open(ctx.image_path).convert("RGB")
            for bbox in ctx.raw_bboxes:
                x1, y1, x2, y2 = bbox
                crop   = img.crop((x1, y1, x2, y2))
                tensor = self._preprocess(crop)
                logits = self._infer("ocr_rec", "x", tensor, "fetch_name_0")[0]
                text, conf = self._ctc_decode(logits)
                ctx.ocr_items.append(OCRItem(text=text, confidence=conf, bbox=bbox))
        except Exception as e:
            ctx.error = f"[RecognizeStage] {e}"
        finally:
            ctx.stage_timings["recognize"] = time.perf_counter() - t
        return ctx

    def _preprocess(self, crop: Image.Image) -> np.ndarray:
        cw, ch = crop.size
        new_w  = max(int(cw * self._rec_height / max(ch, 1) / 4) * 4, 4)
        arr    = np.array(crop.resize((new_w, self._rec_height), Image.BILINEAR), dtype=np.float32) / 255.0
        return ((arr - self.MEAN) / self.STD).transpose(2, 0, 1)[np.newaxis]

    def _ctc_decode(self, logits: np.ndarray) -> tuple[str, float]:
        indices = np.argmax(logits, axis=-1)
        probs   = np.max(logits, axis=-1)
        chars, conf_sum, count, prev = [], 0.0, 0, -1
        for idx, prob in zip(indices, probs):
            if idx != prev:
                if idx != self._blank_idx and self._char_table[idx] is not None:
                    chars.append(self._char_table[idx])
                    conf_sum += float(prob)
                    count += 1
            prev = idx
        return "".join(chars), conf_sum / count if count else 0.0


class FilterStage:
    """신뢰도 임계값 미만 결과 제거 및 정렬."""

    def __init__(self, min_confidence: float = 0.0, sort_by_position: bool = True):
        self._min_conf = min_confidence
        self._sort     = sort_by_position

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        t = time.perf_counter()
        ctx.ocr_items = [i for i in ctx.ocr_items if i.confidence >= self._min_conf]
        if self._sort:
            # 위→아래, 왼→오른 순 정렬
            ctx.ocr_items.sort(key=lambda i: (i.bbox[1], i.bbox[0]))
        ctx.stage_timings["filter"] = time.perf_counter() - t
        return ctx
