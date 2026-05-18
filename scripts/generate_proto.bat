@echo off
REM proto 파일을 현재 환경의 Python으로 재생성합니다.
REM 반드시 서버를 실행할 Python 환경과 동일한 환경에서 실행해야 합니다.

python -m grpc_tools.protoc ^
    -I proto ^
    --python_out=src/generated ^
    --grpc_python_out=src/generated ^
    proto/hello.proto ^
    proto/inference.proto

if errorlevel 1 (
    echo [ERROR] proto 생성 실패. grpcio-tools 설치 여부 확인: pip install grpcio-tools
    exit /b 1
)

REM 생성된 grpc 파일의 import 경로를 패키지 경로로 수정
python -c "
import re, pathlib

for grpc_file in pathlib.Path('src/generated').glob('*_pb2_grpc.py'):
    text = grpc_file.read_text(encoding='utf-8')
    fixed = re.sub(
        r'^import (\w+_pb2) as (\w+)$',
        r'from src.generated import \1 as \2',
        text,
        flags=re.MULTILINE
    )
    if fixed != text:
        grpc_file.write_text(fixed, encoding='utf-8')
        print(f'import 수정 완료: {grpc_file.name}')
"

echo proto 생성 및 import 수정 완료
