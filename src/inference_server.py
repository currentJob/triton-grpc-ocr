import grpc
from concurrent import futures
from src.generated import inference_pb2_grpc
from src.triton_client import TritonClient, TritonConfig
from src.inference_servicer import InferenceServicer


def serve(
    host: str = "0.0.0.0",
    port: int = 50055,
    triton_host: str = "localhost",
    triton_port: int = 8001,
    max_workers: int = 10,
):
    triton_config = TritonConfig(host=triton_host, port=triton_port)
    triton_client = TritonClient(triton_config)

    MB = 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length",    256 * MB),
            ("grpc.max_receive_message_length", 256 * MB),
        ],
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(triton_client), server
    )

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)
    server.start()

    print(f"[InferenceServer] gRPC 추론 게이트웨이 실행 중: {listen_addr}")
    print(f"[InferenceServer] Triton 연결 대상: {triton_host}:{triton_port}")

    try:
        server.wait_for_termination()
    finally:
        triton_client.close()
        print("[InferenceServer] 서버 종료")
