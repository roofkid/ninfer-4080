"""Build the registered Qwen3.8-27B GSQ 3-bit artifact.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_gsq3 \
      --gsq-model /path/to/Qwen3.8-27B-3Bit-GSQ \
      --official-model /path/to/Qwen3.8-27B \
      --out out/qwen3_8_27b_gsq3.ninfer

The quantized body, the vocabulary endpoints, and the Vision tower come from
the pinned GSQ release; the twelve MTP tensors and the frontend resources come
from the official checkpoint.  Fused parents are row selections of the
publisher's packed codes and scales: no code is decoded, requantized, or
rounded, and the only numeric conversion is the audited bf16-to-binary16 scale
width change.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactObject,
    ArtifactWriter,
)
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import convert as family_config
from tools.convert.qwen3_6_27b import draft_head

from . import convert as base_convert
from . import gsq3_source
from . import inventory_gsq3 as inventory
from . import recipe_gsq3 as recipe


RECIPE_ID = "qwen3_8_27b_gsq3-v1"
GSQ_REPOSITORY = gsq3_source.GSQ_REPOSITORY
GSQ_REVISION = gsq3_source.GSQ_REVISION
OFFICIAL_REPOSITORY = gsq3_source.OFFICIAL_REPOSITORY
OFFICIAL_REVISION = gsq3_source.OFFICIAL_REVISION

_QUANTIZATION_CONFIG = {
    "format": "pack-quantized",
    "quant_method": "compressed-tensors",
}
_GROUP_0 = {
    "num_bits": 3,
    "group_size": 128,
    "symmetric": True,
    "strategy": "group",
    "type": "int",
}
_GROUP_1 = {
    "num_bits": 4,
    "group_size": 64,
    "symmetric": True,
    "strategy": "group",
    "type": "int",
}
_GROUP_0_TARGETS = (
    r"re:.*linear_attn.*",
    r"re:.*self_attn.*",
    r"re:.*mlp.*",
)
_IGNORE = (
    r"re:.*visual.*",
    r"re:.*mtp.*",
    r"re:.*linear_attn.*in_proj_a.*",
    r"re:.*linear_attn.*in_proj_b.*",
)

ResourcePayload = family_conversion.ResourcePayload
ObjectPlan = family_conversion.ObjectPlan


@dataclass(frozen=True, slots=True)
class ConversionPreflight:
    gsq_dir: Path
    official_dir: Path
    config_summary: dict[str, object]
    packed_source_count: int
    direct_source: family_recipe.SourcePreflight
    mtp_source: family_recipe.SourcePreflight
    resources: tuple[ResourcePayload, ...]
    draft: draft_head.DraftHeadContext
    object_plan: ObjectPlan


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the official geometry and the pinned GSQ quantization scheme."""

    summary = family_config.validate_config(config)
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping):
        raise ValueError("GSQ config.json has no quantization_config")
    family_conversion.check_members("quantization_config", quantization, _QUANTIZATION_CONFIG)
    if tuple(quantization.get("ignore", ())) != _IGNORE:
        raise ValueError("GSQ quantization_config.ignore does not match the pinned scheme")
    groups = quantization.get("config_groups")
    if not isinstance(groups, Mapping) or set(groups) != {"group_0", "group_1"}:
        raise ValueError("GSQ quantization_config must define exactly group_0 and group_1")
    for name, expected_bits, expected_group, expected_targets in (
        ("group_0", 3, 128, _GROUP_0_TARGETS),
        ("group_1", 4, 64, (r"re:.*embed_tokens.*", r"re:.*lm_head$")),
    ):
        group = groups[name]
        if not isinstance(group, Mapping):
            raise ValueError(f"GSQ quantization_config.{name} is not an object")
        if group.get("input_activations") is not None or group.get("output_activations") is not None:
            raise ValueError(f"GSQ quantization_config.{name} must not quantize activations")
        if tuple(group.get("targets", ())) != expected_targets:
            raise ValueError(f"GSQ quantization_config.{name}.targets does not match the pin")
        weights = group.get("weights")
        if not isinstance(weights, Mapping):
            raise ValueError(f"GSQ quantization_config.{name}.weights is missing")
        expected = dict(_GROUP_0 if expected_bits == 3 else _GROUP_1)
        if expected["group_size"] != expected_group:
            raise ValueError("internal GSQ group expectation mismatch")
        family_conversion.check_members(
            f"quantization_config.{name}.weights", weights, expected
        )
    return summary


def load_frontend_resources(
    gsq_dir: str | Path,
    official_dir: str | Path,
) -> tuple[ResourcePayload, ...]:
    """Load the pinned official frontend set with an official-first fallback.

    The GSQ release republishes five of the six files byte-identically and
    rewrites ``generation_config.json``; the artifact keeps the official words,
    so a missing GSQ copy is not an error and a mismatched candidate is.
    """

    roots = (Path(official_dir), Path(gsq_dir))
    resources: list[ResourcePayload] = []
    for spec in inventory.RESOURCE_SPECS:
        filename = spec.name.removeprefix("frontend/")
        payload = None
        location = None
        for root in roots:
            candidate = root / filename
            if candidate.is_file():
                payload = candidate.read_bytes()
                location = candidate
                break
        if payload is None:
            raise ValueError(
                f"frontend resource {filename} is absent from both source directories"
            )
        if not payload:
            raise ValueError(f"frontend resource {filename} is empty")
        actual = hashlib.sha256(payload).hexdigest()
        expected = base_convert.OFFICIAL_RESOURCE_SHA256[spec.name]
        if actual != expected:
            raise ValueError(
                f"official frontend resource hash mismatch for {location}: "
                f"expected {expected}, got {actual}"
            )
        resources.append(ResourcePayload(spec.name, payload))
    return tuple(resources)


def preflight_inventory() -> None:
    """Establish the one complete target inventory and recipe pairing."""

    if (
        len(inventory.RESOURCE_SPECS),
        len(inventory.TEXT_CORE_TENSOR_SPECS),
        len(inventory.DRAFT_HEAD_TENSOR_SPECS),
        len(inventory.MTP_TENSOR_SPECS),
        len(inventory.VISION_TENSOR_SPECS),
        len(inventory.TENSOR_SPECS),
        len(inventory.OBJECT_SPECS),
    ) != (6, 771, 2, 12, 333, 1118, 1124):
        raise ValueError("registered GSQ3 inventory is incomplete")
    recipe.validate_recipe_coverage()


def build_object_plan(resources: Mapping[str, bytes]) -> ObjectPlan:
    preflight_inventory()
    return family_conversion.build_object_plan(inventory.OBJECT_SPECS, resources)


def open_official_reader(official_dir: Path) -> ShardReader:
    index = official_dir / "model.safetensors.index.json"
    if index.is_file():
        return ShardReader(official_dir)
    shards = sorted(official_dir.glob("*.safetensors"))
    if len(shards) != 1:
        raise ValueError(
            f"{official_dir}: the declared official subset must contain an index or "
            f"exactly one safetensors shard, found {len(shards)}"
        )
    return ShardReader.from_file(shards[0])


def preflight_conversion(
    gsq_model_dir: str | Path,
    official_model_dir: str | Path,
) -> ConversionPreflight:
    """Finish all checkpoint, inventory, shortlist, and offset work before writing."""

    gsq_dir = Path(gsq_model_dir)
    official_dir = Path(official_model_dir)
    config_summary = validate_config(
        family_conversion.load_json(gsq_dir / "config.json")
    )
    preflight_inventory()
    resources = load_frontend_resources(gsq_dir, official_dir)
    object_plan = build_object_plan({resource.name: resource.data for resource in resources})
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    draft = draft_head.compute_shortlist(ranking, gsq_dir)
    with ShardReader(gsq_dir) as gsq_reader, open_official_reader(
        official_dir
    ) as official_reader:
        packed_count, mtp_source, direct_source = recipe.preflight_readers(
            gsq_reader, official_reader
        )
    return ConversionPreflight(
        gsq_dir=gsq_dir,
        official_dir=official_dir,
        config_summary=config_summary,
        packed_source_count=packed_count,
        direct_source=direct_source,
        mtp_source=mtp_source,
        resources=resources,
        draft=draft,
        object_plan=object_plan,
    )


def _derived_tensors(draft: draft_head.DraftHeadContext) -> dict[str, torch.Tensor]:
    return {
        draft_head.DRAFT_HEAD_TOKEN_IDS_OBJECT: (
            draft_head.materialize_draft_head_token_ids(draft)
        )
    }



def materialize_object(
    spec: inventory.TensorSpec,
    gsq_reader: ShardReader,
    official_reader: ShardReader,
    derived: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[bytes, gsq3_source.ScaleAudit | None]:
    """Materialize one artifact payload without requantizing a source code."""

    recipe_spec = recipe.RECIPES_BY_NAME[spec.name]
    expression = recipe_spec.expression
    if isinstance(
        expression,
        (recipe.PackedSource, recipe.PackedRows, recipe.PackedConcat, recipe.PackedGatherRows),
    ):
        matrix = recipe.materialize_packed(expression, gsq_reader, derived)
        if (matrix.rows, matrix.columns) != spec.shape:
            raise ValueError(
                f"{spec.name}: packed shape {(matrix.rows, matrix.columns)} != {spec.shape}"
            )
        payload, audit = gsq3_source.encode_payload(matrix)
        return payload, audit
    sources = _direct_source_names(expression)
    reader = official_reader if sources and all(
        name.startswith("mtp.") for name in sources
    ) else gsq_reader
    tensor = family_recipe.materialize_expression(expression, reader, derived)
    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name}: materialized shape {tuple(tensor.shape)} != {spec.shape}"
        )
    return family_conversion.encode_tensor_payload(tensor, spec, device), None


def _direct_source_names(expression: object) -> tuple[str, ...]:
    return tuple(
        requirement.name
        for requirement in family_recipe.expression_sources(expression)
    )

def _combined_source_preflight(
    preflight: ConversionPreflight,
) -> family_recipe.SourcePreflight:
    direct = preflight.direct_source
    mtp = preflight.mtp_source
    dtypes: Counter[str] = Counter(direct.source_dtype_counts)
    dtypes.update(mtp.source_dtype_counts)
    return family_recipe.SourcePreflight(
        recipe_count=direct.recipe_count + mtp.recipe_count,
        source_tensor_count=(
            direct.source_tensor_count
            + mtp.source_tensor_count
            + preflight.packed_source_count
        ),
        source_shard_count=direct.source_shard_count + mtp.source_shard_count,
        source_dtype_counts={"pack-quantized": preflight.packed_source_count, **dict(dtypes)},
    )


def build_conversion_report(
    *,
    preflight: ConversionPreflight,
    out_path: str | Path,
    arguments: Mapping[str, object],
    objects: Sequence[ArtifactObject],
    elapsed_seconds: float,
    final_bytes: int,
    device: torch.device,
    scale_audit: gsq3_source.ScaleAudit,
) -> dict[str, object]:
    ranking = _repo_root() / draft_head.DEFAULT_RANKING
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=_repo_root(),
        model_dir=preflight.gsq_dir,
        out_path=out_path,
        arguments=arguments,
        config_summary={"base": dict(preflight.config_summary)},
        source_preflight=_combined_source_preflight(preflight),
        objects=objects,
        elapsed_seconds=elapsed_seconds,
        final_bytes=final_bytes,
        device=device,
        ranking_path=ranking,
    )
    report["source"] = {
        "gsq": {
            "repository": GSQ_REPOSITORY,
            "revision": GSQ_REVISION,
            "model_path": str(preflight.gsq_dir.resolve()),
        },
        "official": {
            "repository": OFFICIAL_REPOSITORY,
            "revision": OFFICIAL_REVISION,
            "model_path": str(preflight.official_dir.resolve()),
        },
        "ranking_path": str(ranking.resolve()),
    }
    report["source_preflight"] = {
        "packed": {
            "tensors": preflight.packed_source_count,
        },
        "direct": {
            "recipes": preflight.direct_source.recipe_count,
            "tensors": preflight.direct_source.source_tensor_count,
            "shards": preflight.direct_source.source_shard_count,
            "dtypes": dict(preflight.direct_source.source_dtype_counts),
        },
        "mtp": {
            "recipes": preflight.mtp_source.recipe_count,
            "tensors": preflight.mtp_source.source_tensor_count,
            "shards": preflight.mtp_source.source_shard_count,
            "dtypes": dict(preflight.mtp_source.source_dtype_counts),
        },
    }
    report["scale_conversion"] = scale_audit.as_dict()
    report["scale_conversion"]["note"] = (
        "bf16 multipliers stored as binary16; every rounded word is binary16 "
        "subnormal and bounded by 2**-25"
    )
    return report


def convert(
    gsq_model_dir: str | Path,
    official_model_dir: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    preflight = preflight_conversion(gsq_model_dir, official_model_dir)
    print(
        f"preflight complete: {len(preflight.object_plan.objects)} objects, "
        f"{preflight.packed_source_count} packed sources, "
        f"{preflight.mtp_source.source_tensor_count} MTP sources, "
        f"device={resolved_device}",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    resources = {resource.name: resource.data for resource in preflight.resources}
    derived = _derived_tensors(preflight.draft)
    total = len(inventory.OBJECT_SPECS)
    index = 0
    scale_audit = gsq3_source.ScaleAudit(words=0, rounded=0, max_abs_error=0.0)
    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        preflight.object_plan.specs,
    ) as writer:
        if writer.objects != preflight.object_plan.objects:
            raise RuntimeError("writer object plan differs from completed preflight")
        for spec in inventory.RESOURCE_SPECS:
            index += 1
            writer.write(spec.name, resources[spec.name])
            print(f"[{index}/{total}] {spec.name}", flush=True)
        with ShardReader(preflight.gsq_dir) as gsq_reader, open_official_reader(
            preflight.official_dir
        ) as official_reader:
            for spec in inventory.TENSOR_SPECS:
                index += 1
                payload, audit = materialize_object(
                    spec, gsq_reader, official_reader, derived, resolved_device
                )
                if audit is not None:
                    scale_audit = scale_audit.merge(audit)
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}", flush=True)
    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "gsq_model": str(gsq_model_dir),
        "official_model": str(official_model_dir),
        "out": str(out_path),
        "device": requested_device,
    }
    report = build_conversion_report(
        preflight=preflight,
        out_path=output,
        arguments=arguments,
        objects=preflight.object_plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        scale_audit=scale_audit,
    )
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; "
        f"scale words {scale_audit.words} rounded {scale_audit.rounded}; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsq-model", required=True, type=Path)
    parser.add_argument("--official-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    convert(args.gsq_model, args.official_model, args.out, device=args.device)


if __name__ == "__main__":
    main()
