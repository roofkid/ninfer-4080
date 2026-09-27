"""Persistent-object contract for the Qwen3.8-27B GSQ 3-bit artifact.

The graph, namespace, and non-vocabulary storage roles are identical to the
registered Qwen3.6-27B groupwise artifact.  The only differences are the body
weight format (the source's own 3-bit group-128 grid) and the vocabulary
endpoints (the source's 4-bit group-64 grid, which is also the draft-head
format).
"""

from __future__ import annotations

from dataclasses import replace

from tools.convert.qwen3_6_27b import inventory as qwen3_6_inventory



MODEL_ID = "qwen3.8-27b"
WEIGHTS_ID = "gsq3"
TARGET_KEY = "qwen3_8_27b"

BF16 = qwen3_6_inventory.BF16
FP32 = qwen3_6_inventory.FP32
I32 = qwen3_6_inventory.I32
Q3 = "Q3G128_F16S"
Q4 = qwen3_6_inventory.Q4
Q5 = qwen3_6_inventory.Q5
Q6 = qwen3_6_inventory.Q6
W8 = qwen3_6_inventory.W8

FORMAT_NAMES = (BF16, FP32, I32, Q3, Q4, Q5, Q6, W8)
LAYOUT_NAMES = qwen3_6_inventory.LAYOUT_NAMES
ResourceSpec = qwen3_6_inventory.ResourceSpec
StoredObjectSpec = qwen3_6_inventory.StoredObjectSpec
TensorSpec = qwen3_6_inventory.TensorSpec

FULL_ATTENTION_LAYERS = qwen3_6_inventory.FULL_ATTENTION_LAYERS
GDN_LAYERS = qwen3_6_inventory.GDN_LAYERS
RESOURCE_SPECS = qwen3_6_inventory.RESOURCE_SPECS

# The body matrices the publisher quantized at 3 bits/group 128.  Every other
# text-core object keeps the groupwise target's storage role.
_Q3_SUFFIXES = (
    "attention/output",
    "gdn/output",
    "mlp/down",
)
_Q3_FUSED = (
    "attention/query_key",
    "attention/gate_value",
    "gdn/query_key",
    "gdn/value_z",
    "mlp/gate_up",
)


def _gsq_format(name: str, numeric_format: str) -> str:
    if name in ("text/token_embedding", "text/output_head", "text/draft_head"):
        return Q4
    parts = name.split("/")
    suffix = None
    if len(parts) >= 4 and parts[0] == "text" and parts[1] == "layers":
        suffix = "/".join(parts[3:])
    if suffix is not None and (suffix in _Q3_SUFFIXES or suffix in _Q3_FUSED):
        return Q3
    return numeric_format


TEXT_CORE_TENSOR_SPECS = tuple(
    replace(spec, format=_gsq_format(spec.name, spec.format))
    for spec in qwen3_6_inventory.TEXT_CORE_TENSOR_SPECS
)
DRAFT_HEAD_TENSOR_SPECS = tuple(
    replace(spec, format=Q4) if spec.name == "text/draft_head" else spec
    for spec in qwen3_6_inventory.DRAFT_HEAD_TENSOR_SPECS
)
MTP_TENSOR_SPECS = qwen3_6_inventory.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = qwen3_6_inventory.VISION_TENSOR_SPECS

BASE_TENSOR_SPECS = (
    TEXT_CORE_TENSOR_SPECS
    + DRAFT_HEAD_TENSOR_SPECS
    + MTP_TENSOR_SPECS
    + VISION_TENSOR_SPECS
)
TENSOR_SPECS = BASE_TENSOR_SPECS
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS)
    for layout in LAYOUT_NAMES
}

LOGICAL_ROW_VIEW_SPECS = qwen3_6_inventory.LOGICAL_ROW_VIEW_SPECS
ALIAS_SPECS = qwen3_6_inventory.ALIAS_SPECS


def tensor_specs_by_name() -> dict[str, TensorSpec]:
    return {spec.name: spec for spec in TENSOR_SPECS}


__all__ = [
    "ALIAS_SPECS",
    "BASE_TENSOR_SPECS",
    "BF16",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FORMAT_COUNTS",
    "FORMAT_NAMES",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "LAYOUT_COUNTS",
    "LAYOUT_NAMES",
    "LOGICAL_ROW_VIEW_SPECS",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "OBJECT_SPECS",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "ResourceSpec",
    "StoredObjectSpec",
    "TARGET_KEY",
    "TENSOR_SPECS",
    "TEXT_CORE_TENSOR_SPECS",
    "TensorSpec",
    "VISION_TENSOR_SPECS",
    "W8",
    "WEIGHTS_ID",
    "tensor_specs_by_name",
]
