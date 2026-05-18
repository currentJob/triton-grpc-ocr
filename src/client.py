import grpc
from src.generated import hello_pb2, hello_pb2_grpc


def run():
    with grpc.insecure_channel("localhost:50054") as channel:
        stub = hello_pb2_grpc.HelloServiceStub(channel)

        # Unary 호출
        response = stub.SayHello(hello_pb2.HelloRequest(name="홍길동"))
        print(f"[Client] Unary 응답: {response.message}")

        # Server Streaming 호출
        stream = stub.SayHelloStream(hello_pb2.HelloRequest(name="홍길동"))
        for response in stream:
            print(f"[Client] Stream 응답: {response.message}")

        print("[Client] 스트림 종료")


if __name__ == "__main__":
    run()
