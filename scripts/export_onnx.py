# scripts/export_onnx.py
"""
Export YOLOv8 (.pt) -> ONNX (.onnx)

Why:
- PyTorch(.pt) 모델을 프레임워크 독립적인 ONNX로 내보내면
  다음 단계에서 ONNX Runtime / TensorRT 같은 런타임으로 최적화하기 쉬워짐.

Output:
- outputs/models/yolov8n_320.onnx
"""

from pathlib import Path
from ultralytics import YOLO


def project_root() -> Path:
    """현재 파일 위치 기준으로 프로젝트 루트(=storeguard/)를 찾는다."""
    return Path(__file__).resolve().parents[1]


def export_yolo_to_onnx(
    weights: str = "yolov8n.pt",
    imgsz: int = 320,
    opset: int = 12,
    out_name: str = "yolov8n_320.onnx",
) -> Path:
    """
    YOLO weights(.pt)를 ONNX로 export하고, outputs/models/ 아래로 정리해서 저장한다.

    imgsz:
      - YOLO 입력 해상도(정사각형). 데모/실시간 목적이면 320이 가볍고 빠름.
    opset:
      - ONNX 연산자 버전. 12~13이 호환성이 좋아서 보통 12로 둠.
    """
    root = project_root()
    out_dir = root / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 모델 로드 (필요하면 여기서 자동으로 yolov8n.pt 다운로드됨)
    model = YOLO(weights)

    # 2) ONNX export
    #    Ultralytics는 export 결과 파일 경로를 반환해줌(문자열/경로 형태)
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset)

    exported_path = Path(str(exported))
    target = out_dir / out_name

    # 3) export된 파일이 다른 위치에 생길 수 있으니,
    #    우리가 원하는 outputs/models 로 복사해서 정리
    if exported_path.resolve() != target.resolve():
        target.write_bytes(exported_path.read_bytes())

    return target


def main() -> None:
    onnx_path = export_yolo_to_onnx()

    # 4) 사람이 확인하기 좋은 “스탬프” 출력
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"[OK] ONNX saved: {onnx_path}")
    print(f"[OK] size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
