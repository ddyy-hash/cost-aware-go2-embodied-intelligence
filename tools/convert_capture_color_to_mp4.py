from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from mapping.pc.go2_mapping_calibration.bag import CalibrationCapture
from mapping.pc.go2_mapping_rgbd.decode import decode_image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    writer = None
    with CalibrationCapture(args.capture) as bag:
        for _, payload in bag.iter_messages("color"):
            frame = decode_image(payload)
            image = frame.image
            if frame.encoding.lower() == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif frame.encoding.lower() != "bgr8":
                continue
            if writer is None:
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (frame.width, frame.height))
                if not writer.isOpened():
                    raise RuntimeError(f"unable to open {args.output}")
            writer.write(image)
            count += 1
    if writer is not None:
        writer.release()
    if count == 0:
        raise RuntimeError("no color frames found")
    print(f"MP4_READY frames={count} fps={args.fps:g} path={args.output}")


if __name__ == "__main__":
    main()
