import argparse

def main():
    parser = argparse.ArgumentParser(description="gRPC 서버 실행")
    parser.add_argument(
        "--service",
        choices=["inference", "web"],
        default="inference",
        help="실행할 서비스 (기본값: inference)",
    )
    parser.add_argument(
        "--triton-host",
        default="localhost",
        help="Triton 서버 호스트 (기본값: localhost)",
    )
    parser.add_argument(
        "--triton-port",
        type=int,
        default=8001,
        help="Triton gRPC 포트 (기본값: 8001)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="게이트웨이 gRPC 포트 (inference=50055)",
    )
    args = parser.parse_args()

    if args.service == "inference":
        from src.inference_server import serve
        port = args.port or 50055
        serve(
            port=port,
            triton_host=args.triton_host,
            triton_port=args.triton_port,
        )
    elif args.service == "web":
        from src.web_server import serve
        port = args.port or 8080
        serve(port=port)


if __name__ == "__main__":
    main()
