"""Independent verification for the GSQ-RCO port artifact.

Ported objects are decoded from the artifact and compared against an independent
``gguf-py`` dequantization of their GGUF sources, fused in the same registered
row order.  Copied objects must be byte-identical to the donor GSQ3 artifact.
This is a *second*-generation check: the port re-quantizes already-quantized
values, so the bound is the per-format relative-L2 ceiling below, not the
first-generation source error.  The measured values at the session's conversion
were 0.10-0.13 (Q4G64) and 0.19-0.22 (Q3G128) with the clipping-search grid.

Canonical invocation::

    python -m tools.convert.qwen3_8_27b.verify_gsqrco \
      --artifact out/qwen3_8_27b_gsqrco_iq3s.ninfer \
      --gguf /models/qwen3.8-27b-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
      --source-artifact out/qwen3_8_27b_gsq3.ninfer
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
from tools.artifact.layouts import dequantize_row_split

from . import inventory_gsqrco as inventory
from .gsqrco_source import GgufSource

RELATIVE_L2_BOUNDS = {"Q3G128_F16S": 0.26, "Q4G64_F16S": 0.16, "Q5G64_F16S": 0.09}


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.to(device=actual.device)
    difference = (actual.float() - expected.float()).reshape(-1)
    reference = expected.float().reshape(-1)
    return (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)).item()


def verify(
    artifact_path: str | Path,
    gguf_path: str | Path,
    source_artifact_path: str | Path,
    *,
    sample: int = 0,
    device: str = "cpu",
) -> dict[str, object]:
    started = time.perf_counter()
    source = GgufSource(gguf_path)
    ported = inventory.ported_object_names()
    target = torch.device(device)
    report: dict[str, object] = {
        "artifact": str(artifact_path),
        "gguf": str(gguf_path),
        "source_artifact": str(source_artifact_path),
        "sample": sample,
        "format_counts": {},
        "worst_relative_l2": {},
        "copied_objects": 0,
        "failures": [],
    }
    worst: dict[str, float] = {}
    counts: Counter[str] = Counter()
    checked = 0
    with Artifact.open(artifact_path) as artifact, Artifact.open(source_artifact_path) as donor:
        if artifact.identity.weights_id != inventory.WEIGHTS_ID:
            raise ValueError(f"unexpected weights id {artifact.identity.weights_id}")
        names = [obj.name for obj in artifact.objects if obj.kind == "tensor"]
        ported_names = [name for name in names if name in ported]
        if sample > 0:
            stride = max(1, len(ported_names) // sample)
            ported_names = ported_names[::stride][:sample]
        wanted = set(ported_names)
        for name in names:
            obj = artifact.find(name)
            if name in ported and name not in wanted:
                continue
            if name not in ported:
                donor_obj = donor.find(name)
                if artifact.payload(obj) != donor.payload(donor_obj):
                    report["failures"].append(f"{name}: copied payload differs from the donor")
                report["copied_objects"] = int(report["copied_objects"]) + 1
                continue
            assert obj.kind == "tensor"
            expected = source.object_matrix(name)
            if tuple(obj.shape) != tuple(expected.shape):
                report["failures"].append(
                    f"{name}: artifact shape {obj.shape} != source {tuple(expected.shape)}"
                )
                continue
            actual = dequantize_row_split(
                artifact.payload(obj), obj.format, tuple(obj.shape), device=target,
                dtype=torch.float32,
            )
            if not torch.isfinite(actual).all():
                report["failures"].append(f"{name}: decoded payload contains non-finite values")
                continue
            relative = _relative_l2(actual, expected)
            bound = RELATIVE_L2_BOUNDS.get(obj.format)
            if bound is None or relative > bound:
                report["failures"].append(
                    f"{name}: relative L2 {relative:.4f} exceeds {bound} for {obj.format}"
                )
            counts[obj.format] += 1
            worst[obj.format] = max(worst.get(obj.format, 0.0), relative)
            checked += 1
    report["checked_ported_objects"] = checked
    report["format_counts"] = dict(counts)
    report["worst_relative_l2"] = {key: round(value, 5) for key, value in worst.items()}
    report["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    report["passed"] = not report["failures"]
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path("/models/qwen3.8-27b-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"),
    )
    parser.add_argument("--source-artifact", type=Path, default=Path("out/qwen3_8_27b_gsq3.ninfer"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = verify(
        args.artifact, args.gguf, args.source_artifact, sample=args.sample, device=args.device
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report is not None:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
