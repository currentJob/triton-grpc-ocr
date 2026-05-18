"""Inference 서비스 테스트 클라이언트."""
import numpy as np
import grpc
from src.generated import inference_pb2, inference_pb2_grpc
from src.triton_client import numpy_to_triton_dtype

# 모델별 (input_name, output_names, dummy_shape)
MODEL_TENSORS: dict[str, tuple[str, list[str], tuple]] = {
    "ocr_det":      ("x", ["fetch_name_0"], (1, 3, 640, 640)),
    "ocr_rec":      ("x", ["fetch_name_0"], (1, 3, 48, 320)),
    "doc_ori":      ("x", ["fetch_name_0"], (1, 3, 224, 224)),
    "textline_ori": ("x", ["fetch_name_0"], (1, 3, 80, 160)),
}


def run(host: str = "localhost", port: int = 50055, model_name: str = "ocr_det"):
    MB = 1024 * 1024
    opts = [
        ("grpc.max_send_message_length",    256 * MB),
        ("grpc.max_receive_message_length", 256 * MB),
    ]
    with grpc.insecure_channel(f"{host}:{port}", options=opts) as channel:
        stub = inference_pb2_grpc.InferenceServiceStub(channel)

        # 1. 서버 상태 확인
        health = stub.ServerHealth(inference_pb2.HealthRequest())
        print(f"[Client] 서버 상태  live={health.live}  ready={health.ready}")

        # 2. 모델 준비 상태 확인
        model_status = stub.ModelReady(
            inference_pb2.ModelReadyRequest(model_name=model_name)
        )
        print(f"[Client] 모델 '{model_name}' 준비: {model_status.ready}")

        if not model_status.ready:
            print("[Client] 모델이 준비되지 않아 추론을 건너뜁니다.")
            return

        input_name, output_names, shape = MODEL_TENSORS.get(
            model_name, ("x", ["fetch_name_0"], (1, 3, 224, 224))
        )

        # 3. 단일 추론
        dummy = np.random.randn(*shape).astype(np.float32)
        response = stub.Infer(_build_request(model_name, input_name, dummy, output_names))
        if response.error:
            print(f"[Client] 추론 오류: {response.error}")
        else:
            print(f"[Client] 추론 성공 - 모델: {response.model_name}")
            for out in response.outputs:
                arr = np.frombuffer(out.raw_data, dtype=np.float32).reshape(out.shape)
                print(f"[Client]   출력 '{out.name}': shape={arr.shape}")

        # 4. 스트리밍 추론 (3회)
        print("\n[Client] 스트리밍 추론 (3회)...")
        requests = (
            _build_request(model_name, input_name, np.random.randn(*shape).astype(np.float32), output_names)
            for _ in range(3)
        )
        for i, resp in enumerate(stub.InferStream(requests), 1):
            if resp.error:
                print(f"[Client]   [{i}] 오류: {resp.error}")
            else:
                print(f"[Client]   [{i}] 응답 수신 - 모델: {resp.model_name}")


def _build_request(
    model_name: str,
    input_name: str,
    data: np.ndarray,
    output_names: list[str],
    model_version: str = "",
) -> inference_pb2.InferRequest:
    return inference_pb2.InferRequest(
        model_name=model_name,
        model_version=model_version,
        inputs=[
            inference_pb2.InferInputTensor(
                name=input_name,
                datatype=numpy_to_triton_dtype(data.dtype),
                shape=list(data.shape),
                raw_data=data.tobytes(),
            )
        ],
        outputs=[
            inference_pb2.InferRequestedOutputTensor(name=n) for n in output_names
        ],
    )


if __name__ == "__main__":
    run()
