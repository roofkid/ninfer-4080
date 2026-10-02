"""Convert the published GSQ-RCO IQ3_S GGUF into one complete NInfer artifact.

Only the ported text-core matrix objects are re-encoded: every GGUF tensor is
dequantized with llama.cpp's reference implementation, fused into the registered
artifact row order, and quantized to the format the RCO allocation selected in
:mod:`inventory_gsqrco`.  Vision, DFlash2, the draft head, MTP and every
resource are copied byte-for-byte from the registered GSQ3 artifact, which
carries the same values for those objects, so the port is a controlled test of
the RCO body allocation.

Canonical invocation::

    python -m tools.convert.qwen3_8_27b.convert_gsqrco \
      --gguf /models/qwen3.8-27b-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf \
      --source-artifact out/qwen3_8_27b_gsq3.ninfer \
      --out out/qwen3_8_27b_gsqrco_iq3s.ninfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

from tools.artifact.container import Artifact, ArtifactIdentity, ArtifactWriter
from tools.convert.common.quantize import pick_device
from .gsqrco_quantize import quantize_and_encode
from tools.convert.qwen3_6.common import conversion as family_conversion

from . import inventory_gsqrco as inventory
from .gsqrco_source import GgufSource

RECIPE_ID = "qwen3_8_27b_gsqrco_iq3s-v1"

_DEFAULT_GGUF = Path(
    "/models/qwen3.8-27b-gsq-rco/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf"
)
_DEFAULT_SOURCE_ARTIFACT = Path("out/qwen3_8_27b_gsq3.ninfer")
_DEFAULT_OUT = Path("out/qwen3_8_27b_gsqrco_iq3s.ninfer")
_DEFAULT_UNIFORM_OUT = Path("out/qwen3_8_27b_gsqrco_uniform_q3.ninfer")


def convert(
    gguf_path: str | Path,
    source_artifact_path: str | Path,
    out_path: str | Path,
    *,
    device: str = "cuda",
    uniform_body: bool = False,
) -> Path:
    started = time.perf_counter()
    output = Path(out_path)
    resolved_device = pick_device(device)
    source = GgufSource(gguf_path)
    source_types = source.tensor_types
    specs = inventory.build_object_specs(source_types, uniform_body=uniform_body)
    ported = inventory.ported_object_names()

    format_counts: dict[str, int] = {}
    for spec in specs:
        spec_format = getattr(spec, "format", None)
        if spec_format is not None and spec.name in ported:
            format_counts[spec_format] = format_counts.get(spec_format, 0) + 1

    output.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    encoded = 0
    with Artifact.open(source_artifact_path) as donor:
        resources = {
            spec.name: donor.payload(donor.find(spec.name))
            for spec in inventory.RESOURCE_SPECS
        }
        plan = family_conversion.build_object_plan(specs, resources)
        del resources
        with ArtifactWriter(
            output,
            ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID),
            plan.specs,
        ) as writer:
            if writer.objects != plan.objects:
                raise RuntimeError("writer object plan differs from the preflight plan")
            total = len(writer.objects)
            for index, obj in enumerate(writer.objects, start=1):
                if obj.kind == "tensor" and obj.name in ported:
                    matrix = source.object_matrix(obj.name)
                    payload = quantize_and_encode(matrix, obj.format, device=resolved_device)
                    del matrix
                    encoded += 1
                else:
                    payload = donor.payload(donor.find(obj.name))
                    copied += 1
                writer.write(obj.name, payload)
                del payload
                if index % 64 == 0 or index == total:
                    print(f"[{index}/{total}] {obj.name}", flush=True)
    object_count = total

    elapsed = time.perf_counter() - started
    final_bytes = output.stat().st_size
    report = {
        "recipe_id": RECIPE_ID,
        "model_id": inventory.MODEL_ID,
        "weights_id": inventory.WEIGHTS_ID,
        "gguf": str(gguf_path),
        "source_artifact": str(source_artifact_path),
        "out": str(output),
        "device": str(resolved_device),
        "uniform_body": uniform_body,
        "elapsed_seconds": round(elapsed, 3),
        "final_bytes": final_bytes,
        "object_count": object_count,
        "encoded_objects": encoded,
        "copied_objects": copied,
        "ported_formats": dict(sorted(format_counts.items())),
    }
    report_path = Path(str(output) + ".conversion.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(
        f"complete: {final_bytes} bytes in {elapsed:.1f}s; "
        f"encoded {encoded} copied {copied}; formats {report['ported_formats']}; "
        f"report={report_path}",
        flush=True,
    )
    return report_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=_DEFAULT_GGUF)
    parser.add_argument("--source-artifact", type=Path, default=_DEFAULT_SOURCE_ARTIFACT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--uniform-body",
        action="store_true",
        help="force every ported body object onto the uniform Q3G128 grid (RTN control)",
    )
    args = parser.parse_args(argv)
    out = args.out
    if out is None:
        out = _DEFAULT_UNIFORM_OUT if args.uniform_body else _DEFAULT_OUT
    convert(
        args.gguf,
        args.source_artifact,
        out,
        device=args.device,
        uniform_body=args.uniform_body,
    )


if __name__ == "__main__":
    main()
