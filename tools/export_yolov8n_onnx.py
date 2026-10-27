"""Export YOLOv8n to a static ONNX model for Go2 TensorRT builds."""

from pathlib import Path

from ultralytics import YOLO


def main():
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    exported = Path(
        model.export(
            format="onnx",
            imgsz=640,
            opset=12,
            simplify=False,
            dynamic=False,
        )
    )
    target = output_dir / "yolov8n.onnx"
    target.write_bytes(exported.read_bytes())
    print(target.resolve())


if __name__ == "__main__":
    main()
