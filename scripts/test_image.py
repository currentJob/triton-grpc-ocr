"""이미지 파일을 OCR 파이프라인에 넣는 예시."""
import sys

sys.path.insert(0, ".")
from src.ocr_pipeline import OCRPipeline

CHAR_DICT = "model_repository/_config/charDict.json"


def main(img_path: str):
    pipeline = OCRPipeline(char_dict_path=CHAR_DICT)

    print(f"이미지: {img_path}")
    print("=" * 50)

    results = pipeline.run(img_path)

    if not results:
        print("텍스트 영역이 검출되지 않았습니다.")
        return

    for i, r in enumerate(results, 1):
        x1, y1, x2, y2 = r.bbox
        print(f"[{i:02d}] '{r.text}'")
        print(f"      신뢰도: {r.confidence:.4f}  bbox: ({x1},{y1})-({x2},{y2})")

    print("=" * 50)
    print(f"총 {len(results)}개 텍스트 영역 인식 완료")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python scripts/test_image.py <이미지_경로>")
        print("예시 : python scripts/test_image.py test.jpg")
        sys.exit(1)
    main(sys.argv[1])
