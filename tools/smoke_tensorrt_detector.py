import argparse

import cv2
import numpy as np

from go2_person_follow.detectors import create_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--image", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    detector = create_detector(
        "tensorrt_yolo",
        "",
        args.engine,
        args.input_size,
        args.confidence,
        args.nms,
    )
    if args.image:
        image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {args.image}")
    else:
        image = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = detector.detect(image)
    print(f"detector={getattr(detector, 'backend_name', type(detector).__name__)}")
    print(f"detections={len(detections)}")
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = [float(value) for value in detection.box]
        print(
            "detection_{}=x1:{:.1f},y1:{:.1f},x2:{:.1f},y2:{:.1f},conf:{:.4f}".format(
                index,
                x1,
                y1,
                x2,
                y2,
                detection.confidence,
            )
        )

    if args.output:
        annotated = image.copy()
        for detection in detections:
            raw_x1, raw_y1, raw_x2, raw_y2 = [float(value) for value in detection.box]
            x1 = max(0, int(round(raw_x1)))
            y1 = max(0, int(round(raw_y1)))
            x2 = min(annotated.shape[1] - 1, int(round(raw_x2)))
            y2 = min(annotated.shape[0] - 1, int(round(raw_y2)))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"person {detection.confidence:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(args.output, annotated)


if __name__ == "__main__":
    main()
