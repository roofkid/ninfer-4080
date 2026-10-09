"""Convert the Qwen3.8-27B BF16 checkpoint under the byteshape IQ3_S allocation.

The published byteshape ``Qwen3.8-27B-IQ3_S-3.23bpw.gguf`` contributes only its
per-tensor allocation (the GGML type of every text-core tensor); its values are
not read.  Every text-core matrix is materialized from the official BF16
checkpoint and quantized once with the shared clipping-search encoder
(:mod:`gsqrco_quantize`), so this artifact avoids the second-generation error
of a GGUF-value port.  Vision, MTP and the draft head come from the official
checkpoint recipes, and the DFlash2 companion from the pinned draft release,
exactly as in the other registered identities.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.convert_byteshape \
      --model /models/qwen3.8-27b-bf16 \
      --dflash2-model /models/Qwen3.8-27B-DFlash2 \
      --gguf /models/qwen3.8-27b-byteshape-iq3s/Qwen3.8-27B-IQ3_S-3.23bpw.gguf \
      --out out/qwen3_8_27b_byteshape_iq3s.ninfer
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import torch

from tools.artifact.container import ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6.common.inventory import DIRECT_FORMATS
from tools.convert.qwen3_6_27b import recipe as text_recipe

from . import convert as base_convert
from . import dflash2_recipe
from . import inventory_byteshape as inventory
from .gsqrco_quantize import quantize_and_encode
from .gsqrco_source import gguf_tensor_types

RECIPE_ID = "qwen3_8_27b_byteshape_iq3s-v1"

_DEFAULT_MODEL = Path("/models/qwen3.8-27b-bf16")
_DEFAULT_DFLASH2 = Path("/models/Qwen3.8-27B-DFlash2")
_DEFAULT_GGUF = Path(
    "/models/qwen3.8-27b-byteshape-iq3s/Qwen3.8-27B-IQ3_S-3.23bpw.gguf"
)
_DEFAULT_OUT = Path("out/qwen3_8_27b_byteshape_iq3s.ninfer")


def allocation_digest(source_types: Mapping[str, str]) -> str:
    """Stable digest of the transplanted per-tensor allocation."""

    lines = "".join(f"{name}:{kind}\n" for name, kind in sorted(source_types.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def encode_tensor_payload(
    tensor: torch.Tensor,
    spec: inventory.TensorSpec,
    device: str | torch.device,
) -> bytes:
    """Encode one materialized tensor; quantized formats use the clipping search."""

    if spec.format in DIRECT_FORMATS:
        return family_conversion.encode_tensor_payload(tensor, spec, device)
    return quantize_and_encode(tensor, spec.format, device=device)


def build_object_plan(
    source_types: Mapping[str, str],
    resources: Mapping[str, bytes],
    *,
    uniform_body: bool = False,
) -> family_conversion.ObjectPlan:
    return family_conversion.build_object_plan(
        inventory.build_object_specs(source_types, uniform_body=uniform_body), resources
    )


def convert(
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
    gguf_path: str | Path,
    out_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    uniform_body: bool = False,
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    requested_device = str(device)
    resolved_device = pick_device(device)
    source_types = gguf_tensor_types(gguf_path)
    preflight = base_convert.preflight_conversion(model_dir, dflash2_model_dir)
    base_specs = inventory.build_base_tensor_specs(source_types, uniform_body=uniform_body)
    family_recipe.validate_recipe_coverage(text_recipe.RECIPE_SPECS, base_specs)
    dflash2_recipe.validate_recipe_coverage()
    resources = {resource.name: resource.data for resource in preflight.resources}
    plan = build_object_plan(source_types, resources, uniform_body=uniform_body)
    ported = inventory.ported_object_names()
    format_counts = Counter(
        spec.format for spec in base_specs if spec.name in ported
    )
    print(
        f"preflight complete: {len(plan.objects)} objects, "
        f"{preflight.base_source.source_tensor_count} base and "
        f"{preflight.dflash2_source.source_tensor_count} DFlash2 source tensors, "
        f"allocation={allocation_digest(source_types)[:16]}, "
        f"ported formats={dict(sorted(format_counts.items()))}, device={resolved_device}",
        flush=True,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    total = len(plan.objects)
    index = 0
    with ArtifactWriter(
        output,
        ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        plan.specs,
    ) as writer:
        if writer.objects != plan.objects:
            raise RuntimeError("writer object plan differs from the preflight plan")

        for spec in inventory.RESOURCE_SPECS:
            index += 1
            writer.write(spec.name, resources[spec.name])
            print(f"[{index}/{total}] {spec.name}", flush=True)

        with ShardReader(model_dir) as base_reader:
            for spec in base_specs:
                index += 1
                tensor = base_convert.materialize_tensor(spec, base_reader, preflight.draft)
                payload = encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}", flush=True)

        with ShardReader.from_file(
            preflight.dflash2_model_dir / "model.safetensors"
        ) as dflash2_reader:
            for spec in inventory.DFLASH2_TENSOR_SPECS:
                index += 1
                tensor = dflash2_recipe.materialize_tensor(spec.name, dflash2_reader)
                payload = encode_tensor_payload(tensor, spec, resolved_device)
                del tensor
                writer.write(spec.name, payload)
                del payload
                print(f"[{index}/{total}] {spec.name}", flush=True)

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    arguments = {
        "model": str(model_dir),
        "dflash2_model": str(dflash2_model_dir),
        "gguf": str(gguf_path),
        "out": str(out_path),
        "device": requested_device,
        "uniform_body": uniform_body,
    }
    report = family_conversion.build_conversion_report(
        identity=ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
        target_key=inventory.TARGET_KEY,
        recipe_id=RECIPE_ID,
        repo_root=base_convert._repo_root(),
        model_dir=model_dir,
        out_path=output,
        arguments=arguments,
        config_summary={
            "base": dict(preflight.base_config_summary),
            "dflash2": dict(preflight.dflash2_config_summary),
        },
        source_preflight=preflight.base_source,
        objects=plan.objects,
        elapsed_seconds=elapsed,
        final_bytes=final_bytes,
        device=resolved_device,
        ranking_path=base_convert._repo_root() / base_convert.draft_head.DEFAULT_RANKING,
    )
    type_counts = Counter(source_types.values())
    report["allocation"] = {
        "gguf": str(Path(gguf_path).resolve()),
        "sha256": allocation_digest(source_types),
        "ggml_type_counts": dict(sorted(type_counts.items())),
        "ported_format_counts": dict(sorted(format_counts.items())),
        "uniform_body": uniform_body,
    }
    report["source"] = {
        "base": {
            "repository": "Qwen/Qwen3.8-27B",
            "model_path": str(Path(model_dir).resolve()),
        },
        "dflash2": {
            "repository": dflash2_recipe.REPOSITORY,
            "revision": dflash2_recipe.REVISION,
            "model_path": str(Path(dflash2_model_dir).resolve()),
        },
    }
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; "
        f"formats={dict(sorted(format_counts.items()))}; report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    parser.add_argument("--dflash2-model", type=Path, default=_DEFAULT_DFLASH2)
    parser.add_argument("--gguf", type=Path, default=_DEFAULT_GGUF)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--uniform-body",
        action="store_true",
        help="force every ported body object onto the uniform Q3G128 grid (encoder control)",
    )
    args = parser.parse_args(argv)
    convert(
        args.model,
        args.dflash2_model,
        args.gguf,
        args.out,
        device=args.device,
        uniform_body=args.uniform_body,
    )


if __name__ == "__main__":
    main()
