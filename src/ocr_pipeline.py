"""
PaddleOCR 기반 OCR 파이프라인
  1. ocr_det  : 텍스트 영역 검출 (DB 알고리즘)
  2. ocr_rec  : 텍스트 인식 (CTC 디코딩)
"""
import json
import numpy as np
import grpc
from dataclasses import dataclass
from PIL import Image

from src.generated import inference_pb2, inference_pb2_grpc


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2


class OCRPipeline:
    DET_MODEL   = "ocr_det"
    REC_MODEL   = "ocr_rec"
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        char_dict_path: str,
        gateway_host: str = "localhost",
        gateway_port: int = 50055,
        det_thresh: float = 0.3,
        rec_height: int = 48,
    ):
        self.det_thresh = det_thresh
        self.rec_height = rec_height
        self._load_char_dict(char_dict_path)

        MB = 1024 * 1024
        channel = grpc.insecure_channel(
            f"{gateway_host}:{gateway_port}",
            options=[
                ("grpc.max_send_message_length",    256 * MB),
                ("grpc.max_receive_message_length", 256 * MB),
            ],
        )
        self._stub = inference_pb2_grpc.InferenceServiceStub(channel)

    # ── public ─────────────────────────────────────────────────────────────

    def run(self, img_path: str) -> list[OCRResult]:
        img = Image.open(img_path).convert("RGB")
        bboxes = self._detect(img)
        results = [self._recognize(img, bbox) for bbox in bboxes]
        return results

    # ── detection ──────────────────────────────────────────────────────────

    def _detect(self, img: Image.Image) -> list[tuple[int, int, int, int]]:
        orig_w, orig_h = img.size
        x, scale_h, scale_w = self._preprocess_det(img)
        prob_map = self._infer(self.DET_MODEL, "x", x, "fetch_name_0")[0, 0]  # H x W

        binary = (prob_map > self.det_thresh).astype(np.uint8)
        bboxes_norm = self._extract_bboxes(binary)

        # 모델 입력 좌표 → 원본 이미지 좌표로 변환
        bboxes = []
        for x1, y1, x2, y2 in bboxes_norm:
            bboxes.append((
                max(0, int(x1 / scale_w)),
                max(0, int(y1 / scale_h)),
                min(orig_w, int(x2 / scale_w)),
                min(orig_h, int(y2 / scale_h)),
            ))
        return bboxes

    def _preprocess_det(
        self, img: Image.Image, max_side: int = 960
    ) -> tuple[np.ndarray, float, float]:
        orig_w, orig_h = img.size
        scale = min(max_side / max(orig_h, orig_w), 1.0)
        new_h = max(int(orig_h * scale / 32) * 32, 32)
        new_w = max(int(orig_w * scale / 32) * 32, 32)
        resized = img.resize((new_w, new_h), Image.BILINEAR)

        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD
        arr = arr.transpose(2, 0, 1)[np.newaxis]   # 1CHW
        return arr, new_h / orig_h, new_w / orig_w

    @staticmethod
    def _extract_bboxes(
        binary: np.ndarray, min_area: int = 100
    ) -> list[tuple[int, int, int, int]]:
        """연결된 텍스트 영역을 bounding box로 변환 (scipy 없이 구현)."""
        # 수평 팽창으로 가까운 글자 연결
        from itertools import groupby
        h, w = binary.shape
        dilated = np.zeros_like(binary)
        for r in range(h):
            row = binary[r]
            # 수평으로 8픽셀 팽창
            for c in range(w):
                if row[c]:
                    dilated[r, max(0, c-8):min(w, c+9)] = 1

        visited = np.zeros_like(dilated, dtype=bool)
        bboxes = []

        for r in range(h):
            for c in range(w):
                if dilated[r, c] and not visited[r, c]:
                    # BFS로 연결 컴포넌트 탐색
                    queue = [(r, c)]
                    visited[r, c] = True
                    min_r, max_r, min_c, max_c = r, r, c, c
                    while queue:
                        cr, cc = queue.pop()
                        min_r = min(min_r, cr)
                        max_r = max(max_r, cr)
                        min_c = min(min_c, cc)
                        max_c = max(max_c, cc)
                        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<h and 0<=nc<w and dilated[nr,nc] and not visited[nr,nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
                    area = (max_r - min_r + 1) * (max_c - min_c + 1)
                    if area >= min_area:
                        bboxes.append((min_c, min_r, max_c, max_r))
        return bboxes

    # ── recognition ────────────────────────────────────────────────────────

    def _recognize(
        self, img: Image.Image, bbox: tuple[int, int, int, int]
    ) -> OCRResult:
        x1, y1, x2, y2 = bbox
        crop = img.crop((x1, y1, x2, y2))
        x = self._preprocess_rec(crop)
        logits = self._infer(self.REC_MODEL, "x", x, "fetch_name_0")[0]  # seq_len x 11947
        text, conf = self._ctc_decode(logits)
        return OCRResult(text=text, confidence=conf, bbox=bbox)

    def _preprocess_rec(self, crop: Image.Image) -> np.ndarray:
        cw, ch = crop.size
        new_w = max(int(cw * self.rec_height / ch / 4) * 4, 4)
        resized = crop.resize((new_w, self.rec_height), Image.BILINEAR)
        arr = np.array(resized, dtype=np.float32) / 255.0
        arr = (arr - self.MEAN) / self.STD
        arr = arr.transpose(2, 0, 1)[np.newaxis]   # 1 x 3 x 48 x W
        return arr

    def _ctc_decode(
        self, logits: np.ndarray
    ) -> tuple[str, float]:
        """CTC greedy decode: argmax → 연속 중복 제거 → blank 제거."""
        indices = np.argmax(logits, axis=-1)   # seq_len
        probs   = np.max(logits, axis=-1)      # seq_len (before softmax, use as proxy)

        chars, conf_sum, count = [], 0.0, 0
        prev = -1
        for idx, prob in zip(indices, probs):
            if idx != prev:
                if idx != self._blank_idx and self._char_table[idx] is not None:
                    chars.append(self._char_table[idx])
                    conf_sum += float(prob)
                    count += 1
            prev = idx

        text = "".join(chars)
        confidence = conf_sum / count if count > 0 else 0.0
        return text, confidence

    # ── gRPC helper ────────────────────────────────────────────────────────

    def _infer(
        self, model: str, input_name: str, data: np.ndarray, output_name: str
    ) -> np.ndarray:
        req = inference_pb2.InferRequest(
            model_name=model,
            inputs=[inference_pb2.InferInputTensor(
                name=input_name,
                datatype="FP32",
                shape=list(data.shape),
                raw_data=data.tobytes(),
            )],
            outputs=[inference_pb2.InferRequestedOutputTensor(name=output_name)],
        )
        resp = self._stub.Infer(req)
        if resp.error:
            raise RuntimeError(f"[{model}] {resp.error}")
        out = resp.outputs[0]
        return np.frombuffer(out.raw_data, dtype=np.float32).reshape(out.shape)

    # ── char dict ──────────────────────────────────────────────────────────

    def _load_char_dict(self, path: str):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        self._char_table: list = d["charTable"]
        self._blank_idx: int   = d["blankIdx"]
