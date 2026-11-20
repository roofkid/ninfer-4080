"""Convert the release SenseNova-Loop-9B BF16 checkpoint into a reference artifact.

The canonical artifact is produced by ``awq.py`` from the official AWQ checkpoint.
Reference invocation::

    python -m tools.convert.sensenova_loop_9b.convert \
      --model /models/SenseNova-Loop-9B \
      --out out/sensenova_loop_9b_bf16_reference.ninfer
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactObject, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import inventory, recipe


RECIPE_ID = "sensenova_loop_9b-groupwise-int-v1"

_ROOT_CONFIG = {
    "architectures": ["SenseNovaLoopForConditionalGeneration"],
    "model_type": "sensenova_loop",
    "tie_word_embeddings": False,
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
}
_LOOP_CONFIG = {
    "start": 4,
    "end": 28,
    "counts": [2],
    "eval_count": 2,
}
_MHC_CONFIG = {
    "start": 4,
    "end": 28,
    "streams": 2,
    "iterations": 20,
    "fixed_identity": False,
    "pre_scale": 1.0,
}
_TEXT_CONFIG = {
    "model_type": "qwen3_5_text",
    "num_hidden_layers": 32,
    "full_attention_interval": 4,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "attention_bias": False,
    "attn_output_gate": True,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
    "tie_word_embeddings": False,
    "max_position_embeddings": 524288,
    "rms_norm_eps": 1e-6,
    "partial_rotary_factor": 0.25,
}
_ROPE_CONFIG = {
    "rope_theta": 10000000,
    "rope_type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 262144,
    "partial_rotary_factor": 0.25,
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
}
_VISION_CONFIG = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "out_hidden_size": 4096,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 2304,
    "hidden_act": "gelu_pytorch_tanh",
}

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan
SourcePreflight = recipe.SourcePreflight


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    model_dir: Path
    config_summary: dict[str, object]
    source: SourcePreflight
    resources: tuple[ResourcePayload, ...]
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_config(model_dir: Path) -> dict[str, object]:
    return family_conversion.load_json(model_dir / "config.json")


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    family_conversion.check_members("config", config, _ROOT_CONFIG)
    family_conversion.check_members("loop_settings", config["loop_settings"], _LOOP_CONFIG)
    family_conversion.check_members("mhc", config["mhc"], _MHC_CONFIG)
    text = config.get("text_config")
    vision = config.get("vision_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")
    family_conversion.check_members("text_config", text, _TEXT_CONFIG)
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    family_conversion.check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    family_conversion.check_members("vision_config", vision, _VISION_CONFIG)
    layer_types = text.get("layer_types")
    expected = [
        "full_attention" if layer in inventory.FULL_ATTENTION_LAYERS else "linear_attention"
        for layer in range(inventory.PHYSICAL_LAYERS)
    ]
    if layer_types != expected:
        raise ValueError("text_config.layer_types does not match the registered 32-layer schedule")
    return {
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "text": {name: text[name] for name in _TEXT_CONFIG},
        "loop": dict(_LOOP_CONFIG),
        "mhc": dict(_MHC_CONFIG),
        "layer_types": {
            "physical_layers": inventory.PHYSICAL_LAYERS,
            "execution_slots": inventory.EXECUTION_SLOTS,
            "full_attention_layers": list(inventory.FULL_ATTENTION_LAYERS),
            "gdn_layers": list(inventory.GDN_LAYERS),
            "mhc_layers": list(inventory.MHC_LAYERS),
        },
        "rope": {name: rope[name] for name in _ROPE_CONFIG},
        "vision": {name: vision[name] for name in _VISION_CONFIG},
        "vision_token_ids": {
            name: config[name]
            for name in (
                "vision_start_token_id",
                "vision_end_token_id",
                "image_token_id",
                "video_token_id",
            )
        },
    }


def preflight_inventory() -> None:
    if len(inventory.RESOURCE_SPECS) != 6:
        raise ValueError("registered frontend resource inventory is incomplete")
    if len(inventory.FULL_ATTENTION_LAYERS) != 8 or len(inventory.GDN_LAYERS) != 24:
        raise ValueError("registered physical topology is incomplete")
    if len(inventory.MHC_LAYERS) != 24:
        raise ValueError("registered mHC range is incomplete")
    if len(inventory.MTP_TENSOR_SPECS) != 12:
        raise ValueError("registered MTP inventory is incomplete")
    if len(inventory.VISION_TENSOR_SPECS) != 333:
        raise ValueError("registered Vision inventory is incomplete")
    recipe.validate_recipe_coverage()


def load_resources(model_dir: str | Path) -> tuple[ResourcePayload, ...]:
    return family_conversion.load_resources(model_dir, inventory.RESOURCE_SPECS)


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def _merge_preflight(main: SourcePreflight, mtp: SourcePreflight) -> SourcePreflight:
    dtypes = dict(main.source_dtype_counts)
    for dtype, count in mtp.source_dtype_counts.items():
        dtypes[dtype] = dtypes.get(dtype, 0) + count
    return SourcePreflight(
        recipe_count=main.recipe_count + mtp.recipe_count,
        source_tensor_count=main.source_tensor_count + mtp.source_tensor_count,
        source_shard_count=main.source_shard_count + mtp.source_shard_count,
        source_dtype_counts=dtypes,
    )


def preflight_conversion(model_dir: str | Path) -> ConversionPreflight:
    model = Path(model_dir)
    config_summary = validate_config(_load_config(model))
    preflight_inventory()
    source = _merge_preflight(recipe.preflight_main(model), recipe.preflight_mtp(model))
    resources = load_resources(model)
    resource_map = {resource.name: resource.data for resource in resources}
    return ConversionPreflight(
        model_dir=model,
        config_summary=config_summary,
        source=source,
        resources=resources,
        object_plan=build_object_plan(resource_map),
    )


def materialize_tensor(
    spec: inventory.TensorSpec,
    main_reader: ShardReader,
    mtp_reader: ShardReader,
) -> torch.Tensor:
    source_reader = mtp_reader if spec.name in inventory.MTP_OBJECT_NAMES else main_reader
    tensor = recipe.materialize_recipe(recipe.RECIPES_BY_NAME[spec.name], source_reader)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return tensor


def build_conversion_report(
    *,
    model_dir: str | Path,
    out_path: str | Path,
    arguments: Mapping[str, object],
    config_summary: Mapping[str, object],
    source_preflight: SourcePreflight,
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    revision: str | None = None,
) -> dict[str, object]:
    return family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=model_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary=config_summary,
        source_preflight=source_preflight,
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=model_dir,
        revision=revision,
    )


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(model)

    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.source.source_tensor_count} source tensors, device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    mtp_path = model / "mtp.safetensors"
    with ShardReader(model) as main_reader, ShardReader.from_file(mtp_path) as mtp_reader:
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            preflight.object_plan.specs,
        ) as writer:
            if writer.objects != preflight.object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resources[spec.name]
                else:
                    tensor = materialize_tensor(spec, main_reader, mtp_reader)
                    payload = family_conversion.encode_tensor_payload(tensor, spec, resolved_device)
                    del tensor
                writer.write(spec.name, payload)
                del payload
                if index % 32 == 0 or index == len(inventory.OBJECT_SPECS):
                    print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report = build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments={
            "model": str(model_dir),
            "out": str(out_path),
            "device": requested_device,
        },
        config_summary=preflight.config_summary,
        source_preflight=preflight.source,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"complete: {final_bytes} bytes in {elapsed:.1f}s; report={report_path}", flush=True)
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.model, args.out, device=args.device)


if __name__ == "__main__":
    main()
