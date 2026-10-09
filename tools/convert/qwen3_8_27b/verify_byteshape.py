"""Independent verification for the Qwen3.8-27B byteshape-allocation artifact.

Every tensor is decoded from the artifact and compared against the registered
BF16/DFlash2 source it was encoded from.  Direct formats must reproduce the
source words exactly; quantized formats must stay inside the per-format
relative-L2 bound.  The quantizer itself is covered by the shared FP64 oracle
tests, so this check covers the artifact plan, layouts, writer and source
materialization rather than re-deriving the quantization.

Canonical invocation::

    python3 -m tools.convert.qwen3_8_27b.verify_byteshape \
      --artifact out/qwen3_8_27b_byteshape_iq3s.ninfer \
      --model /models/qwen3.8-27b-bf16 \
      --dflash2-model /models/Qwen3.8-27B-DFlash2 \
      --gguf /models/qwen3.8-27b-byteshape-iq3s/Qwen3.8-27B-IQ3_S-3.23bpw.gguf
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Sequence

import torch

from tools.artifact.container import Artifact
from tools.artifact.layouts import decode_direct, dequantize_row_split
from tools.convert.common.safetensors import ShardReader

from . import convert as base_convert
from . import convert_byteshape
from . import dflash2_recipe
from . import inventory_byteshape as inventory
from .gsqrco_source import gguf_tensor_types

# First-generation bounds: these values are quantized once from BF16, so the
# second-generation port bounds (Q3 0.26, Q4 0.16, Q5 0.09) do not apply.
RELATIVE_L2_BOUNDS = {
    "Q3G128_F16S": 0.24,
    "Q4G64_F16S": 0.16,
    "Q5G64_F16S": 0.08,
    "Q6G64_F16S": 0.06,
    "W8G32_F16S": 0.04,
}
DIRECT_FORMATS = ("BF16", "FP32", "I32")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.float() - expected.float()).reshape(-1)
    reference = expected.float().reshape(-1)
    return (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)).item()


def verify(
    artifact_path: str | Path,
    model_dir: str | Path,
    dflash2_model_dir: str | Path,
    gguf_path: str | Path,
    *,
    sample: int = 0,
    device: str = "cpu",
) -> dict[str, object]:
    started = time.perf_counter()
    source_types = gguf_tensor_types(gguf_path)
    base_specs = inventory.build_base_tensor_specs(source_types)
    specs = {spec.name: spec for spec in base_specs}
    specs.update({spec.name: spec for spec in inventory.DFLASH2_TENSOR_SPECS})
    dflash2_specs = {spec.name for spec in inventory.DFLASH2_TENSOR_SPECS}
    preflight = base_convert.preflight_conversion(model_dir, dflash2_model_dir)
    target = torch.device(device)
    report: dict[str, object] = {
        "artifact": str(artifact_path),
        "gguf": str(gguf_path),
        "allocation_sha256": convert_byteshape.allocation_digest(source_types),
        "sample": sample,
        "checked_tensors": 0,
        "format_counts": {},
        "worst_relative_l2": {},
        "failures": [],
    }
    worst: dict[str, float] = {}
    counts: Counter[str] = Counter()

    with Artifact.open(artifact_path) as artifact:
        if artifact.identity.weights_id != inventory.WEIGHTS_ID:
            raise ValueError(f"unexpected weights id {artifact.identity.weights_id}")
        tensor_names = [obj.name for obj in artifact.objects if obj.kind == "tensor"]
        expected_names = [spec.name for spec in base_specs] + [
            spec.name for spec in inventory.DFLASH2_TENSOR_SPECS
        ]
        if tensor_names != expected_names:
            raise ValueError("artifact tensor order does not match the byteshape inventory")
        wanted = set(tensor_names)
        if sample > 0:
            stride = max(1, len(tensor_names) // sample)
            wanted = set(tensor_names[::stride][:sample])

        for spec in inventory.RESOURCE_SPECS:
            obj = artifact.find(spec.name)
            expected_resource = next(
                resource for resource in preflight.resources if resource.name == spec.name
            )
            if artifact.payload(obj) != expected_resource.data:
                report["failures"].append(f"{spec.name}: resource payload differs from source")

        with ShardReader(model_dir) as base_reader, ShardReader.from_file(
            dflash2_model_dir / "model.safetensors"
        ) as dflash2_reader:
            for name in tensor_names:
                if name not in wanted:
                    continue
                spec = specs[name]
                obj = artifact.find(name)
                if obj.format != spec.format or tuple(obj.shape) != tuple(spec.shape):
                    report["failures"].append(
                        f"{name}: artifact {obj.format}/{tuple(obj.shape)} != plan "
                        f"{spec.format}/{tuple(spec.shape)}"
                    )
                    continue
                if name in dflash2_specs:
                    expected = dflash2_recipe.materialize_tensor(name, dflash2_reader)
                else:
                    expected = base_convert.materialize_tensor(
                        spec, base_reader, preflight.draft
                    )
                payload = artifact.payload(obj)
                if obj.format in DIRECT_FORMATS:
                    actual = decode_direct(payload, obj.format, obj.shape, device=target)
                    if not torch.equal(actual.float(), expected.float()):
                        report["failures"].append(f"{name}: direct words differ from source")
                    relative = 0.0
                else:
                    actual = dequantize_row_split(
                        payload, obj.format, obj.shape, device=target, dtype=torch.float32
                    )
                    if not torch.isfinite(actual).all():
                        report["failures"].append(f"{name}: decoded values are not finite")
                        continue
                    relative = _relative_l2(actual, expected)
                    bound = RELATIVE_L2_BOUNDS[obj.format]
                    if relative > bound:
                        report["failures"].append(
                            f"{name}: relative L2 {relative:.4f} exceeds {bound} "
                            f"for {obj.format}"
                        )
                counts[obj.format] += 1
                worst[obj.format] = max(worst.get(obj.format, 0.0), relative)
                del actual, expected, payload
                if int(report["checked_tensors"]) % 64 == 0:
                    print(
                        f"[{report['checked_tensors']}/{len(wanted)}] {name}",
                        flush=True,
                    )
                report["checked_tensors"] = int(report["checked_tensors"]) + 1

    report["format_counts"] = dict(sorted(counts.items()))
    report["worst_relative_l2"] = {key: round(value, 5) for key, value in worst.items()}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    report["passed"] = not report["failures"]
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/models/qwen3.8-27b-bf16"))
    parser.add_argument(
        "--dflash2-model", type=Path, default=Path("/models/Qwen3.8-27B-DFlash2")
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path(
            "/models/qwen3.8-27b-byteshape-iq3s/Qwen3.8-27B-IQ3_S-3.23bpw.gguf"
        ),
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = verify(
        args.artifact,
        args.model,
        args.dflash2_model,
        args.gguf,
        sample=args.sample,
        device=args.device,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
