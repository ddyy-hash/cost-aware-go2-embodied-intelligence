"""Run one local MobileVLA-R1 inference without network or robot control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=str(ROOT / "models" / "mobilevla-r1-download" / "extracted" / "weight" / "sft"),
    )
    parser.add_argument("--variant", choices=("sft", "rl"), default="rl")
    parser.add_argument("--source-root", default=str(ROOT / "third_party" / "MobileVLA-R1"))
    parser.add_argument("--manifest", default=str(ROOT / "mapping" / "config" / "mobilevla_r1_manifest.json"))
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--load-in-4bit", action="store_true", help="load the local checkpoint with NF4 bitsandbytes quantization")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default=str(ROOT / "output" / "assets" / "mobilevla-inference-result.json"))
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(ROOT))
    from mapping.pc.go2_mapping_assets import verify_tree_identity
    from mapping.pc.go2_mapping_vla.offline import classify_model_response
    from mapping.pc.go2_mapping_vla.official import OfficialMobileVLABackend

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    default_model_path = ROOT / "models" / "mobilevla-r1-download" / "extracted" / "weight" / "sft"
    if Path(args.model_path) == default_model_path:
        args.model_path = str(default_model_path.parent / args.variant)
    identity_key = "local_sft" if args.variant == "sft" else "local_rl"
    verify_tree_identity(
        args.model_path,
        digest=str(manifest[f"{identity_key}_directory_sha256"]),
        file_count=int(manifest[f"{identity_key}_file_count"]),
        total_bytes=int(manifest[f"{identity_key}_total_bytes"]),
    )

    started = time.time()
    backend = OfficialMobileVLABackend(
        model_path=args.model_path,
        source_root=args.source_root,
        device=args.device,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        load_in_4bit=args.load_in_4bit,
        expected_identity={
            "directory_sha256": manifest[f"{identity_key}_directory_sha256"],
            "file_count": manifest[f"{identity_key}_file_count"],
            "total_bytes": manifest[f"{identity_key}_total_bytes"],
        },
        variant=args.variant,
    )
    backend.load()
    response = backend.infer(args.image, args.instruction)
    result = {
        "schema_version": 1,
        "model_id": f"MobileVLA-R1-{args.variant.upper()}",
        "variant": args.variant,
        "model_path": str(Path(args.model_path).resolve()),
        "source_root": str(Path(args.source_root).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint_directory_sha256": manifest[f"{identity_key}_directory_sha256"],
        "image_paths": [str(Path(path).resolve()) for path in args.image],
        "instruction": args.instruction,
        "response": response,
        "structured_response": classify_model_response(response),
        "local_files_only": True,
        "action_boundary": "DIAGNOSTIC_TEXT_ONLY",
        "navigation_activation": False,
        "elapsed_seconds": round(time.time() - started, 3),
        "load_in_4bit": args.load_in_4bit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print("MOBILEVLA_INFERENCE_RESULT " + json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
