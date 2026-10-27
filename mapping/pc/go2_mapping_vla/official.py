"""Local MobileVLA-R1 inference backend.

This module keeps the upstream model loader isolated from the Dashboard. It
loads only local files and returns the upstream text/action observation for the
safe semantic adapter; it never publishes robot commands.
"""
from __future__ import annotations

import importlib.machinery
import os
from pathlib import Path
import sys
from typing import Any

from mapping.pc.go2_mapping_assets import verify_tree_identity


def model_name_for_variant(variant: str) -> str:
    """Return the upstream loader identity for a staged checkpoint variant."""
    if variant not in {"sft", "rl"}:
        raise ValueError("MobileVLA variant must be sft or rl")
    return f"mobilevla-r1-{variant}"


def build_navigation_prompt(instruction: str, *, frame_count: int = 1) -> str:
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be non-empty")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    historical_tokens = "<image>\n" * max(frame_count - 1, 0)
    return (
        "Imagine you are a robot programmed for navigation tasks. You have "
        "been given a video of historical observations and a local current "
        "observation "
        f"{historical_tokens}and current observation <image>\n. "
        f"Your assigned task is: \"{instruction}\" "
        "Analyze this series of images to decide your next action, which "
        "could be inspecting a room, inspecting a feature, exploring a "
        "frontier, going to a named room, or requesting an operator stop. "
        "Return exactly one JSON object with the keys \"intent\", \"target\", "
        "\"confidence\", \"reasoning_summary\", and \"constraints\". "
        "The intent must be one of inspect_room, inspect_feature, "
        "explore_frontiers, go_to_room, or stop_and_request_operator. "
        "The constraints object must include "
        "\"require_operator_confirmation\": true for operator confirmation. "
        "The caller will validate "
        "this JSON object before any navigation. Do not output velocity, "
        "joint, gait, footstep, or motor commands. "
    )


def _install_optional_import_shims() -> None:
    """Allow the upstream inference import without its training-only extras."""
    if "deepspeed" not in sys.modules:
        deepspeed = type(sys)("deepspeed")
        deepspeed.__spec__ = importlib.machinery.ModuleSpec("deepspeed", None)
        comm = type(sys)("deepspeed.comm")
        comm.__spec__ = importlib.machinery.ModuleSpec("deepspeed.comm", None)
        comm.is_initialized = lambda: False
        comm.get_rank = lambda: 0
        comm.get_world_size = lambda: 1
        deepspeed.comm = comm
        sys.modules["deepspeed"] = deepspeed
        sys.modules["deepspeed.comm"] = comm
    if "s2wrapper" not in sys.modules:
        s2wrapper = type(sys)("s2wrapper")
        s2wrapper.__spec__ = importlib.machinery.ModuleSpec("s2wrapper", None)
        s2wrapper.forward = lambda *args, **kwargs: None
        sys.modules["s2wrapper"] = s2wrapper
    if "flash_attn" not in sys.modules:
        flash_attn = type(sys)("flash_attn")
        flash_attn.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
        flash_attn.__path__ = []
        interface = type(sys)("flash_attn.flash_attn_interface")
        interface.__spec__ = importlib.machinery.ModuleSpec("flash_attn.flash_attn_interface", None)
        interface.flash_attn_unpadded_qkvpacked_func = lambda *args, **kwargs: None
        interface.flash_attn_varlen_qkvpacked_func = lambda *args, **kwargs: None
        padding = type(sys)("flash_attn.bert_padding")
        padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", None)
        padding.pad_input = lambda *args, **kwargs: None
        padding.unpad_input = lambda *args, **kwargs: None
        flash_attn.flash_attn_interface = interface
        flash_attn.bert_padding = padding
        sys.modules["flash_attn"] = flash_attn
        sys.modules["flash_attn.flash_attn_interface"] = interface
        sys.modules["flash_attn.bert_padding"] = padding


class OfficialMobileVLABackend:
    """Load and run the official local MobileVLA checkpoint."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        source_root: str | Path,
        device: str = "cuda",
        device_map: str = "auto",
        max_new_tokens: int = 128,
        load_in_4bit: bool = False,
        expected_identity: dict[str, Any] | None = None,
        variant: str = "sft",
    ) -> None:
        self.model_path = Path(model_path)
        self.source_root = Path(source_root)
        self.device = device
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        self.expected_identity = expected_identity
        self.variant = variant
        self.model_name = model_name_for_variant(variant)
        self._tokenizer: Any = None
        self._model: Any = None
        self._image_processor: Any = None

    def load(self) -> None:
        if not self.source_root.is_dir() or not self.model_path.is_dir():
            raise RuntimeError("local MobileVLA source/model directory is unavailable")
        if self.expected_identity is not None:
            verify_tree_identity(
                self.model_path,
                digest=str(self.expected_identity["directory_sha256"]),
                file_count=int(self.expected_identity["file_count"]),
                total_bytes=int(self.expected_identity["total_bytes"]),
            )
        _install_optional_import_shims()
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        # The upstream loader is intentionally called with local paths only.
        previous = {key: os.environ.get(key) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from llava.model.builder import load_pretrained_model

            loader_options = {"load_4bit": self.load_in_4bit}
            if self.load_in_4bit:
                import torch
                from transformers import BitsAndBytesConfig
                loader_options["torch_dtype"] = torch.float16
                loader_options["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                # The vendored loader adds the legacy load_in_4bit kwarg when
                # load_4bit=True; Transformers rejects that alongside an
                # explicit quantization_config, so pass only the config.
                loader_options["load_4bit"] = False
            self._tokenizer, self._model, self._image_processor, _ = load_pretrained_model(
                str(self.model_path),
                self.model_name,
                device=self.device,
                device_map=self.device_map,
                **loader_options,
            )
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def infer(self, image_paths: list[str | Path], instruction: str) -> str:
        if not self.loaded:
            self.load()
        if not image_paths:
            raise ValueError("at least one local image is required")
        import torch
        from PIL import Image
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.conversation import SeparatorStyle, conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        images = [Image.open(path).convert("RGB") for path in image_paths]
        question = build_navigation_prompt(instruction, frame_count=len(images))
        conversation = conv_templates["llama_3"].copy()
        conversation.append_message(conversation.roles[0], question)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        model_device = getattr(self._model, "device", None) or next(self._model.parameters()).device
        vision_dtype = next(self._model.get_vision_tower().parameters()).dtype
        image_tensor = process_images(images, self._image_processor, self._model.config).to(
            model_device, dtype=vision_dtype
        )
        input_ids = tokenizer_image_token(
            prompt, self._tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model_device)
        pad_token_id = self._tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self._tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_token_id)
        stop = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2
        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
        result = self._tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if result.endswith(stop):
            result = result[: -len(stop)].strip()
        return result
