import numpy as np
import tritonclient.grpc as grpcclient
from dataclasses import dataclass, field
from typing import Optional

# Triton 데이터 타입 <-> numpy dtype 매핑
_TRITON_TO_NUMPY: dict[str, type] = {
    "FP16": np.float16,
    "FP32": np.float32,
    "FP64": np.float64,
    "INT8": np.int8,
    "INT16": np.int16,
    "INT32": np.int32,
    "INT64": np.int64,
    "UINT8": np.uint8,
    "UINT16": np.uint16,
    "UINT32": np.uint32,
    "UINT64": np.uint64,
    "BOOL": np.bool_,
}

_NUMPY_TO_TRITON: dict[type, str] = {v: k for k, v in _TRITON_TO_NUMPY.items()}


@dataclass
class TritonConfig:
    host: str = "localhost"
    port: int = 8001
    ssl: bool = False
    timeout: float = 30.0
    headers: dict[str, str] = field(default_factory=dict)


class TritonClient:
    """Triton Inference Server gRPC 클라이언트 래퍼."""

    def __init__(self, config: TritonConfig):
        url = f"{config.host}:{config.port}"
        self._client = grpcclient.InferenceServerClient(
            url=url,
            ssl=config.ssl,
        )
        self._timeout = config.timeout
        self._headers = config.headers or None

    def is_alive(self) -> bool:
        try:
            return self._client.is_server_live(
                headers=self._headers, client_timeout=self._timeout
            )
        except Exception:
            return False

    def is_ready(self) -> bool:
        try:
            return self._client.is_server_ready(
                headers=self._headers, client_timeout=self._timeout
            )
        except Exception:
            return False

    def is_model_ready(self, model_name: str, model_version: str = "") -> bool:
        try:
            return self._client.is_model_ready(
                model_name,
                model_version,
                headers=self._headers,
                client_timeout=self._timeout,
            )
        except Exception:
            return False

    def infer(
        self,
        model_name: str,
        inputs_data: dict[str, np.ndarray],
        output_names: list[str],
        model_version: str = "",
        datatype_override: Optional[dict[str, str]] = None,
    ) -> dict[str, np.ndarray]:
        """Triton에 추론 요청을 보내고 numpy 배열 결과를 반환합니다."""
        inputs = []
        for name, data in inputs_data.items():
            dtype_str = (
                datatype_override.get(name)
                if datatype_override
                else None
            ) or numpy_to_triton_dtype(data.dtype)
            infer_input = grpcclient.InferInput(name, list(data.shape), dtype_str)
            infer_input.set_data_from_numpy(data)
            inputs.append(infer_input)

        outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

        results = self._client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            model_version=model_version,
            headers=self._headers,
            client_timeout=self._timeout,
        )

        return {name: results.as_numpy(name) for name in output_names}

    def close(self):
        self._client.close()


def triton_to_numpy_dtype(datatype: str) -> type:
    return _TRITON_TO_NUMPY.get(datatype, np.float32)


def numpy_to_triton_dtype(dtype: np.dtype) -> str:
    return _NUMPY_TO_TRITON.get(dtype.type, "BYTES")
