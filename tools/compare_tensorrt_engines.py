import argparse
import statistics
import time
from pathlib import Path

import cv2

from go2_person_follow.detectors import create_detector


def find_images(image_dir: Path, limit: int):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in suffixes)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"no images found in {image_dir}")
    return paths


def evaluate_engine(name: str, engine_path: str, image_paths, input_size: int):
    detector = create_detector("tensorrt_yolo", "", engine_path, input_size, 0.30, 0.45)
    latencies_ms = []
    detection_counts = []
    max_confidences = []

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {path}")
        start_s = time.perf_counter()
        detections = detector.detect(image)
        latencies_ms.append((time.perf_counter() - start_s) * 1000.0)
        detection_counts.append(len(detections))
        max_confidences.append(max((det.confidence for det in detections), default=0.0))

    print(f"{name}_engine={engine_path}")
    print(f"{name}_images={len(image_paths)}")
    print(f"{name}_mean_latency_ms={statistics.mean(latencies_ms):.3f}")
    print(f"{name}_median_latency_ms={statistics.median(latencies_ms):.3f}")
    print(f"{name}_total_detections={sum(detection_counts)}")
    print(f"{name}_max_confidence={max(max_confidences):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--uncalibrated-engine", required=True)
    parser.add_argument("--calibrated-engine", required=True)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--limit", type=int, default=32)
    args = parser.parse_args()

    image_paths = find_images(Path(args.image_dir), args.limit)
    evaluate_engine("uncalibrated", args.uncalibrated_engine, image_paths, args.input_size)
    evaluate_engine("calibrated", args.calibrated_engine, image_paths, args.input_size)


if __name__ == "__main__":
    main()
