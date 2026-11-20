"""Convert the official SenseNova-Loop-9B AWQ checkpoint into one .ninfer artifact.

The official AWQ release stores the text matrices as 4-bit GEMM-packed AWQ
tensors and the token embedding as a row-major 4-bit table.  This reader
reconstructs the logical dense tensors, including the AWQ-adjusted RMSNorm
weights, and feeds the existing groupwise-int recipe instead of adding an AWQ
runtime format.

Canonical invocation::

    python -m tools.convert.sensenova_loop_9b.awq \
      --model /models/SenseNova-Loop-9B-AWQ \
      --out out/sensenova_loop_9b.ninfer \
      --resource-fallback /models/SenseNova-Loop-9B \
      --source-revision 8327226ffd2e33e34cb5466a0e6cccd4419bde40
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader, TensorMetadata
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import inventory, recipe
from . import convert as bf16_convert


RECIPE_ID = "sensenova_loop_9b-awq-groupwise-int-v1"

_AWQ_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)
_AWQ_SOURCE_DTYPE = "AWQ4"

_ROOT_CONFIG = {
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
    "mtp_num_hidden_layers": 0,
    "mtp_use_dedicated_embeddings": False,
    "tie_word_embeddings": False,
    "max_position_embeddings": 262144,
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
_QUANTIZATION_CONFIG = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "version": "gemm",
    "lm_head": True,
}


@dataclass(frozen=True, slots=True)
class QuantizedTensor:
    logical_name: str
    layout: str
    qweight: str
    qzeros: str
    scales: str


def _unpack(qweight: torch.Tensor, order: Sequence[int] | None) -> torch.Tensor:
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qweight.device)
    words = (qweight.unsqueeze(-1) >> shifts) & 0xF
    if order is not None:
        words = words[..., list(order)]
    return words.reshape(qweight.shape[0], -1)


def _dequantize(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    *,
    layout: str,
    group_size: int | None = None,
) -> torch.Tensor:
    """Reconstruct an AWQ GEMM linear or row-major embedding tensor.

    GEMM weights are logical ``[out, in]``; the row-major embedding is
    ``[vocab, hidden]``.  Zero points are stored directly (no GPTQ -1 offset).
    """

    if layout == "gemm":
        rows, packed_cols = qweight.shape
        cols = packed_cols * 8
        groups = scales.shape[0]
        actual_group = rows // groups
        if group_size is not None and actual_group != group_size:
            raise ValueError(f"AWQ group size {actual_group} != declared {group_size}")
        codes = _unpack(qweight, _AWQ_ORDER).to(torch.float32)
        zeros = _unpack(qzeros, _AWQ_ORDER).to(torch.float32)
        values = codes.reshape(groups, actual_group, cols)
        zeros = zeros.reshape(groups, 1, cols)
        scales = scales.to(torch.float32).reshape(groups, 1, cols)
        return ((values - zeros) * scales).reshape(rows, cols).T.contiguous()

    if layout == "row_major":
        vocab, packed_cols = qweight.shape
        cols = packed_cols * 8
        groups = scales.shape[1]
        actual_group = cols // groups
        if group_size is not None and actual_group != group_size:
            raise ValueError(f"AWQ group size {actual_group} != declared {group_size}")
        codes = _unpack(qweight, None).to(torch.float32)
        zeros = _unpack(qzeros, None).to(torch.float32)
        values = codes.reshape(vocab, groups, actual_group)
        zeros = zeros.reshape(vocab, groups, 1)
        scales = scales.to(torch.float32).reshape(vocab, groups, 1)
        return ((values - zeros) * scales).reshape(vocab, cols)

    raise ValueError(f"unsupported AWQ layout {layout!r}")


def dequantize_gemm(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Public oracle for a GEMM-layout AWQ linear tensor."""

    return _dequantize(qweight, qzeros, scales, layout="gemm", group_size=group_size)


def dequantize_row_major(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Public oracle for the row-major quantized token embedding."""

    return _dequantize(qweight, qzeros, scales, layout="row_major", group_size=group_size)


class AwqReader:
    """Expose AWQ checkpoint tensors under their logical dense names."""

    def __init__(
        self,
        model_dir: str | Path,
        index_filename: str = "model.safetensors.index.json",
    ) -> None:
        self.model_dir = Path(model_dir)
        index = json.loads((self.model_dir / index_filename).read_text())
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._reader = ShardReader(self.model_dir, index_filename)
        self._quantized: dict[str, QuantizedTensor] = {}
        for name in self.weight_map:
            if not name.endswith(".qweight"):
                continue
            prefix = name[: -len(".qweight")]
            logical = prefix + ".weight"
            qzeros = prefix + ".qzeros"
            scales = prefix + ".scales"
            missing = [key for key in (qzeros, scales) if key not in self.weight_map]
            if missing:
                raise ValueError(f"{prefix}: missing AWQ tensors {missing}")
            layout = "row_major" if prefix.endswith("embed_tokens") else "gemm"
            self._quantized[logical] = QuantizedTensor(
                logical_name=logical,
                layout=layout,
                qweight=name,
                qzeros=qzeros,
                scales=scales,
            )
        if not self._quantized:
            raise ValueError(f"{self.model_dir}: AWQ index contains no quantized tensors")

    @property
    def names(self) -> tuple[str, ...]:
        direct = (
            name
            for name in self.weight_map
            if not name.endswith((".qweight", ".qzeros", ".scales"))
        )
        return tuple(self._quantized) + tuple(direct)

    def has(self, name: str) -> bool:
        return name in self._quantized or name in self.weight_map

    def metadata(self, names: Iterable[str]) -> dict[str, TensorMetadata]:
        requested = tuple(dict.fromkeys(names))
        physical = [
            self._quantized[name].qweight if name in self._quantized else name
            for name in requested
        ]
        physical_metadata = self._reader.metadata(physical)
        result: dict[str, TensorMetadata] = {}
        for name in requested:
            quantized = self._quantized.get(name)
            if quantized is None:
                result[name] = physical_metadata[name]
                continue
            qmeta = physical_metadata[quantized.qweight]
            qweight_shape = qmeta.shape
            if quantized.layout == "gemm":
                shape = (qweight_shape[1] * 8, qweight_shape[0])
            else:
                shape = (qweight_shape[0], qweight_shape[1] * 8)
            result[name] = TensorMetadata(
                name=name,
                shard=qmeta.shard,
                shape=shape,
                dtype=_AWQ_SOURCE_DTYPE,
            )
        return result

    def get(self, name: str) -> torch.Tensor:
        quantized = self._quantized.get(name)
        if quantized is None:
            return self._reader.get(name)

        qweight = self._reader.get(quantized.qweight)
        qzeros = self._reader.get(quantized.qzeros)
        scales = self._reader.get(quantized.scales)
        if scales.dtype != torch.float16:
            raise ValueError(f"{quantized.logical_name}: expected F16 scales, got {scales.dtype}")
        return _dequantize(
            qweight,
            qzeros,
            scales,
            layout=quantized.layout,
            group_size=_QUANTIZATION_CONFIG["group_size"],
        )

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> AwqReader:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def open_mtp_reader(model_dir: str | Path) -> ShardReader:
    model = Path(model_dir)
    index = model / "mtp" / "model.safetensors.index.json"
    if index.is_file():
        return ShardReader.from_index(index)
    mtp = model / "mtp.safetensors"
    if mtp.is_file():
        return ShardReader.from_file(mtp)
    raise FileNotFoundError(f"{model}: no AWQ MTP index or mtp.safetensors")


def _load_config(model_dir: Path) -> dict[str, object]:
    return family_conversion.load_json(model_dir / "config.json")


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    family_conversion.check_members("config", config, _ROOT_CONFIG)
    family_conversion.check_members("loop_settings", config["loop_settings"], _LOOP_CONFIG)
    family_conversion.check_members("mhc", config["mhc"], _MHC_CONFIG)
    text = config.get("text_config")
    vision = config.get("vision_config")
    quantization = config.get("quantization_config")
    if not isinstance(text, Mapping) or not isinstance(vision, Mapping):
        raise ValueError("config.json must contain text_config and vision_config")
    if not isinstance(quantization, Mapping):
        raise ValueError("config.json must contain quantization_config")
    family_conversion.check_members("text_config", text, _TEXT_CONFIG)
    rope = text.get("rope_parameters")
    if not isinstance(rope, Mapping):
        raise ValueError("text_config.rope_parameters is missing")
    family_conversion.check_members("text_config.rope_parameters", rope, _ROPE_CONFIG)
    family_conversion.check_members("vision_config", vision, _VISION_CONFIG)
    family_conversion.check_members("quantization_config", quantization, _QUANTIZATION_CONFIG)
    embedding = quantization.get("embedding")
    if not isinstance(embedding, Mapping):
        raise ValueError("quantization_config.embedding is missing")
    family_conversion.check_members(
        "quantization_config.embedding",
        embedding,
        {"bits": 4, "group_size": 128, "layout": "row_major", "zero_point": True},
    )
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
        "quantization": {
            "method": quantization["quant_method"],
            "bits": quantization["bits"],
            "group_size": quantization["group_size"],
            "zero_point": quantization["zero_point"],
            "version": quantization["version"],
            "lm_head": quantization["lm_head"],
            "embedding": dict(embedding),
        },
    }


def preflight_reader(
    reader: AwqReader | ShardReader,
    recipes: Sequence[recipe.TensorRecipe],
) -> recipe.SourcePreflight:
    requirements = recipe.source_requirements_for(recipes)
    metadata = reader.metadata(requirements)
    dtype_counts: dict[str, int] = {}
    shards: set[str] = set()
    for name, requirement in requirements.items():
        actual = metadata[name]
        if actual.shape != requirement.shape:
            raise ValueError(f"{name}: source shape {actual.shape} != required {requirement.shape}")
        dtype_counts[actual.dtype] = dtype_counts.get(actual.dtype, 0) + 1
        shards.add(actual.shard)
    return recipe.SourcePreflight(
        recipe_count=len(recipes),
        source_tensor_count=len(requirements),
        source_shard_count=len(shards),
        source_dtype_counts=dtype_counts,
    )


def _merge_preflight(
    main: recipe.SourcePreflight,
    mtp: recipe.SourcePreflight,
) -> recipe.SourcePreflight:
    dtypes = dict(main.source_dtype_counts)
    for dtype, count in mtp.source_dtype_counts.items():
        dtypes[dtype] = dtypes.get(dtype, 0) + count
    return recipe.SourcePreflight(
        recipe_count=main.recipe_count + mtp.recipe_count,
        source_tensor_count=main.source_tensor_count + mtp.source_tensor_count,
        source_shard_count=main.source_shard_count + mtp.source_shard_count,
        source_dtype_counts=dtypes,
    )


def load_resources(
    model_dir: str | Path,
    fallback_dir: str | Path | None = None,
) -> tuple[family_conversion.ResourcePayload, ...]:
    root = Path(model_dir)
    fallback = None if fallback_dir is None else Path(fallback_dir)
    resources: list[family_conversion.ResourcePayload] = []
    for spec in inventory.RESOURCE_SPECS:
        filename = spec.name.removeprefix("frontend/")
        path = root / filename
        if not path.is_file() and fallback is not None:
            path = fallback / filename
        if not path.is_file():
            raise FileNotFoundError(f"frontend resource {path} is missing")
        data = path.read_bytes()
        if not data:
            raise ValueError(f"frontend resource {path} is empty")
        resources.append(family_conversion.ResourcePayload(spec.name, data))
    return tuple(resources)


_DIRECT_DTYPES = {
    "BF16": torch.bfloat16,
    "FP32": torch.float32,
    "I32": torch.int32,
}


def _cast_direct(tensor: torch.Tensor, spec: inventory.TensorSpec) -> torch.Tensor:
    if spec.format not in _DIRECT_DTYPES:
        return tensor
    dtype = _DIRECT_DTYPES[spec.format]
    return tensor if tensor.dtype == dtype else tensor.to(dtype)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    resource_fallback: str | Path | None = None,
    source_revision: str | None = None,
) -> Path:
    started = time.perf_counter()
    model = Path(model_dir)
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    config_summary = validate_config(_load_config(model))
    bf16_convert.preflight_inventory()

    with AwqReader(model) as main_reader, open_mtp_reader(model) as mtp_reader:
        main_preflight = preflight_reader(main_reader, recipe.MAIN_RECIPES)
        mtp_preflight = preflight_reader(mtp_reader, recipe.MTP_RECIPES)
        source_preflight = _merge_preflight(main_preflight, mtp_preflight)
        resources = load_resources(model, resource_fallback)
        resource_map = {resource.name: resource.data for resource in resources}
        object_plan = bf16_convert.build_object_plan(resource_map)
        print(
            f"preflight complete: {len(object_plan.objects)} objects, "
            f"{source_preflight.source_tensor_count} source tensors, "
            f"device={resolved_device}",
            flush=True,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            object_plan.specs,
        ) as writer:
            if writer.objects != object_plan.objects:
                raise RuntimeError("writer object plan differs from completed preflight")
            for index, spec in enumerate(inventory.OBJECT_SPECS, start=1):
                if isinstance(spec, inventory.ResourceSpec):
                    payload = resource_map[spec.name]
                else:
                    tensor = bf16_convert.materialize_tensor(spec, main_reader, mtp_reader)
                    tensor = _cast_direct(tensor, spec)
                    payload = family_conversion.encode_tensor_payload(
                        tensor, spec, resolved_device
                    )
                    del tensor
                writer.write(spec.name, payload)
                del payload
                if index % 32 == 0 or index == len(inventory.OBJECT_SPECS):
                    print(f"[{index}/{len(inventory.OBJECT_SPECS)}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report = bf16_convert.build_conversion_report(
        model_dir=model,
        out_path=output,
        arguments={
            "model": str(model_dir),
            "out": str(out_path),
            "device": requested_device,
            "source_format": "awq-gemm",
            "source_revision": source_revision,
            "resource_fallback": (
                None if resource_fallback is None else str(Path(resource_fallback))
            ),
        },
        config_summary=config_summary,
        source_preflight=source_preflight,
        objects=object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        revision=None,
    )
    report["recipe_id"] = RECIPE_ID
    report["source"]["format"] = "awq-gemm"
    report["source"]["revision"] = source_revision
    report["source"]["config_sha256"] = _sha256(model / "config.json")
    report["source"]["index_sha256"] = _sha256(model / "model.safetensors.index.json")
    report["source"]["mtp_index_sha256"] = _sha256(model / "mtp" / "model.safetensors.index.json")
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
    parser.add_argument("--resource-fallback", type=Path)
    parser.add_argument("--source-revision")
    args = parser.parse_args(argv)
    convert(
        args.model,
        args.out,
        device=args.device,
        resource_fallback=args.resource_fallback,
        source_revision=args.source_revision,
    )


if __name__ == "__main__":
    main()
