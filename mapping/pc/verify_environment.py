import json
import sys


def main() -> int:
    import safetensors
    import torch
    import transformers

    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }
    if not result["cuda_available"]:
        print(json.dumps(result, indent=2))
        return 2

    result["gpu"] = torch.cuda.get_device_name(0)
    result["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    tensor = torch.ones((1024, 1024), device="cuda")
    result["cuda_tensor_sum"] = float(tensor.sum().item())

    import gsplat  # noqa: F401
    import open3d  # noqa: F401
    import gtsam  # noqa: F401
    import nerfstudio  # noqa: F401

    result["status"] = "ok"
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
