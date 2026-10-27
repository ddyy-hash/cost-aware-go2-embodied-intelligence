import argparse
import ctypes
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import tensorrt as trt


class CudaRuntime:
    MEMCPY_HOST_TO_DEVICE = 1

    def __init__(self):
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        if ptr:
            self._check(self.lib.cudaFree(ptr))

    def copy_host_to_device(self, dst: ctypes.c_void_p, src: np.ndarray) -> None:
        self._check(
            self.lib.cudaMemcpy(
                dst,
                src.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(src.nbytes),
                ctypes.c_int(self.MEMCPY_HOST_TO_DEVICE),
            )
        )

    @staticmethod
    def _check(code: int) -> None:
        if int(code) != 0:
            raise RuntimeError(f"CUDA runtime call failed with code {code}")


def preprocess_image(path: Path, input_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {path}")

    height, width = image.shape[:2]
    scale = min(float(input_size) / float(width), float(input_size) / float(height))
    resized_w = int(round(float(width) * scale))
    resized_h = int(round(float(height) * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - resized_w) // 2
    pad_y = (input_size - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) * (1.0 / 255.0)
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor)


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: Sequence[Path],
        input_shape: Tuple[int, int, int, int],
        input_size: int,
        cache_file: Path,
    ):
        super().__init__()
        self.image_paths = list(image_paths)
        self.input_shape = input_shape
        self.batch_size = int(input_shape[0])
        self.input_size = int(input_size)
        self.cache_file = Path(cache_file)
        self.index = 0
        self.cuda = CudaRuntime()
        self.host_batch = np.empty(input_shape, dtype=np.float32)
        self.device_input = self.cuda.malloc(self.host_batch.nbytes)

    def __del__(self):
        try:
            self.cuda.free(self.device_input)
        except Exception:
            pass

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: List[str]):
        if self.index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self.index : self.index + self.batch_size]
        if len(batch_paths) < self.batch_size:
            return None

        for offset, path in enumerate(batch_paths):
            self.host_batch[offset] = preprocess_image(path, self.input_size)

        self.cuda.copy_host_to_device(self.device_input, self.host_batch)
        self.index += self.batch_size
        print(f"calibrated_images={self.index}/{len(self.image_paths)}")
        return [int(self.device_input.value)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)


def find_images(image_dir: Path, limit: int) -> List[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in suffixes)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"no calibration images found in {image_dir}")
    return paths


def build_engine(args) -> None:
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    onnx_bytes = Path(args.onnx).read_bytes()
    if not parser.parse(onnx_bytes):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError("failed to parse ONNX model")

    config = builder.create_builder_config()
    config.max_workspace_size = int(args.workspace_mib) * 1024 * 1024
    config.set_flag(trt.BuilderFlag.INT8)
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    input_shape = tuple(int(dim) for dim in input_tensor.shape)
    if any(dim <= 0 for dim in input_shape):
        input_shape = (int(args.batch_size), 3, int(args.input_size), int(args.input_size))
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)
    else:
        input_shape = (int(input_shape[0]), int(input_shape[1]), int(input_shape[2]), int(input_shape[3]))

    if input_shape[0] != int(args.batch_size):
        print(f"using model batch size {input_shape[0]} instead of requested {args.batch_size}")

    image_paths = find_images(Path(args.calib_dir), int(args.max_images))
    usable_count = (len(image_paths) // input_shape[0]) * input_shape[0]
    image_paths = image_paths[:usable_count]
    if not image_paths:
        raise RuntimeError("not enough calibration images for one batch")

    calibrator = ImageEntropyCalibrator(
        image_paths,
        input_shape,
        int(args.input_size),
        Path(args.cache),
    )
    config.int8_calibrator = calibrator

    print(f"onnx={args.onnx}")
    print(f"engine={args.engine}")
    print(f"cache={args.cache}")
    print(f"calibration_images={len(image_paths)}")
    print(f"input_shape={input_shape}")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("failed to build TensorRT engine")
        serialized_engine = engine.serialize()

    Path(args.engine).parent.mkdir(parents=True, exist_ok=True)
    Path(args.engine).write_bytes(serialized_engine)
    print(f"saved_engine={args.engine}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--calib-dir", required=True)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=128)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()
