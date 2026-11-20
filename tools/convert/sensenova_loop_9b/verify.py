"""Independent structural and payload verification for a SenseNova-Loop-9B artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from tools.artifact.container import Artifact, ResourceObject, TensorObject
from tools.artifact.layouts import (
    dequantize_row_split,
    encode_direct,
    gather_row_planes,
    row_split_geometry,
)
from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.common.safetensors import ShardReader
from . import inventory, recipe

DIRECT_FORMATS = inventory.DIRECT_FORMATS


class VerificationError(ValueError):
    """The artifact does not match the registered target contract."""


def _fail(message: str) -> None:
    raise VerificationError(message)


def verify_structure(artifact: Artifact) -> None:
    if artifact.identity.model_id != inventory.MODEL_ID:
        _fail(f"model identity {artifact.identity.model_id!r} is not {inventory.MODEL_ID!r}")
    if artifact.identity.weights_id != inventory.WEIGHTS_ID:
        _fail(
            f"weights identity {artifact.identity.weights_id!r} is not {inventory.WEIGHTS_ID!r}"
        )
    expected = tuple(inventory.OBJECT_SPECS)
    actual = tuple(artifact.objects)
    if len(expected) != len(actual):
        _fail(f"artifact has {len(actual)} objects, expected {len(expected)}")
    for spec, obj in zip(expected, actual):
        if spec.name != obj.name:
            _fail(f"object order mismatch: {obj.name!r} != {spec.name!r}")
        if isinstance(spec, inventory.ResourceSpec):
            if not isinstance(obj, ResourceObject) or obj.encoding != spec.encoding:
                _fail(f"{spec.name}: resource identity mismatch")
        else:
            if not isinstance(obj, TensorObject):
                _fail(f"{spec.name}: expected a tensor object")
            if tuple(obj.shape) != spec.shape:
                _fail(f"{spec.name}: shape {tuple(obj.shape)} != {spec.shape}")
            if obj.format != spec.format or obj.layout != spec.layout:
                _fail(f"{spec.name}: storage signature mismatch")


def _source_reader_for(name: str, main, mtp):
    return mtp if name in inventory.MTP_OBJECT_NAMES else main


def _source_tensor(name: str, main, mtp) -> torch.Tensor:
    reader = _source_reader_for(name, main, mtp)
    return recipe.materialize_recipe(recipe.RECIPES_BY_NAME[name], reader)


_DIRECT_DTYPES = {
    "BF16": torch.bfloat16,
    "FP32": torch.float32,
    "I32": torch.int32,
}


def verify_direct_objects(artifact: Artifact, main, mtp) -> int:
    count = 0
    for spec in inventory.TENSOR_SPECS:
        if spec.format not in DIRECT_FORMATS:
            continue
        source = _source_tensor(spec.name, main, mtp)
        expected = encode_direct(source.to(_DIRECT_DTYPES[spec.format]), spec.format)
        if expected != bytes(artifact.payload(spec.name)):
            _fail(f"{spec.name}: direct payload differs from the source")
        count += 1
    return count


QUANT_PROBES = (
    "text/layers/3/attention/query_key",
    "text/layers/0/gdn/value_z",
    "text/layers/0/gdn/query_key",
    "text/layers/4/mlp/gate_up",
    "mtp/layer/attention/query_key_gate_value",
)
PROBE_ROWS = 3


def _probe_rows(rows: int) -> list[int]:
    if rows <= PROBE_ROWS:
        return list(range(rows))
    return sorted({0, rows // 2, rows - 1})


def verify_quantized_objects(artifact: Artifact, main, mtp) -> int:
    checked = 0
    for name in QUANT_PROBES:
        obj = artifact.find(name)
        spec = next(spec for spec in inventory.TENSOR_SPECS if spec.name == name)
        if spec.format in DIRECT_FORMATS:
            continue
        rows = _probe_rows(spec.shape[0])
        geometry = row_split_geometry(spec.format, spec.shape)
        planes = gather_row_planes(artifact.payload(name), geometry, rows)
        decoded = dequantize_row_split(
            planes, spec.format, (len(rows), spec.shape[1]), dtype=torch.float32
        )
        source = _source_tensor(name, main, mtp).index_select(
            0, torch.tensor(rows, dtype=torch.long)
        )
        expect = source.float().numpy()
        actual = decoded[: len(rows), : spec.shape[1]].numpy()
        if not np.isfinite(actual).all():
            _fail(f"{name}: decoded probe contains NaN or infinity")
        format_spec = get_format(spec.format)
        if not isinstance(format_spec, QuantFormat):
            _fail(f"{name}: expected a quantized format")
        group = format_spec.group_size
        expected_groups = expect.reshape(len(rows), -1, group)
        error = np.abs(actual - expect).reshape(expected_groups.shape)
        bound = np.max(np.abs(expected_groups), axis=-1) / float(format_spec.qmax)
        if not np.all(error <= bound[:, :, None] * 1.01 + 1.0e-3):
            _fail(f"{name}: decoded quantized probe exceeds its group step")
        checked += 1
    return checked


def _open_sources(model: Path, source: str):
    if source == "awq":
        from .awq import AwqReader, open_mtp_reader

        return AwqReader(model), open_mtp_reader(model)
    if source == "bf16":
        return ShardReader(model), ShardReader.from_file(model / "mtp.safetensors")
    raise ValueError(f"unknown source format {source!r}")


def verify_artifact(
    path: str | Path,
    model_dir: str | Path,
    *,
    source: str = "awq",
) -> dict[str, object]:
    model = Path(model_dir)
    main, mtp = _open_sources(model, source)
    with Artifact.open(path) as artifact, main, mtp:
        verify_structure(artifact)
        direct = verify_direct_objects(artifact, main, mtp)
        quantized = verify_quantized_objects(artifact, main, mtp)
    return {
        "path": str(path),
        "source": source,
        "objects": len(inventory.OBJECT_SPECS),
        "direct_objects": direct,
        "quantized_probes": quantized,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", choices=("awq", "bf16"), default="awq")
    args = parser.parse_args(argv)
    summary = verify_artifact(args.artifact, args.model, source=args.source)
    print(
        f"verified {summary['objects']} objects from {summary['source']} source "
        f"({summary['direct_objects']} direct, {summary['quantized_probes']} quantized probes)"
    )


if __name__ == "__main__":
    main()
