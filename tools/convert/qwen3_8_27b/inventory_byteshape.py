"""Persistent-object contract for the Qwen3.8-27B byteshape IQ3_S port.

The text-core matrix formats follow the published per-tensor allocation of
``byteshape/Qwen3.8-27B-GGUF`` ``Qwen3.8-27B-IQ3_S-3.23bpw.gguf``.  That
release labels its files with size classes, not llama.cpp profiles: IQ3_XXS is
the workhorse and IQ4_XS/Q5_K/Q6_K cover selected attention, MLP, vocabulary
and GDN roles.  The GGML names map onto registered formats as

    IQ3_XXS  -> Q3G128_F16S
    IQ2_XXS  -> Q3G128_F16S   (no registered 2-bit family; promoted)
    IQ4_XS   -> Q4G64_F16S
    Q5_K     -> Q5G64_F16S
    Q6_K     -> Q6G64_F16S    (rank 6; fused sites still resolve to Q4/Q5)
    Q8_0     -> W8G32_F16S    (GDN control projections; not in the ported set)

Only the allocation is transplanted: values are quantized from the official
BF16 checkpoint, never read from the GGUF.  Fused objects take the widest
contributor because one artifact object carries one numeric format, and the
fused-pair routes are exactly those the ops admit (shared with
:mod:`inventory_gsqrco`).  The vocabulary endpoints keep the registered Q4G64
grid, and Vision, MTP, the draft head and the DFlash2 companion are the
registered first-generation objects, byte-identical to the GSQ3 plan.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from . import inventory_gsq3
from . import inventory_gsqrco as route_map

from tools.convert.qwen3_6.common.inventory import StoredObjectSpec, TensorSpec

MODEL_ID = inventory_gsq3.MODEL_ID
WEIGHTS_ID = "byteshape-iq3s"
TARGET_KEY = inventory_gsq3.TARGET_KEY

BF16 = inventory_gsq3.BF16
FP32 = inventory_gsq3.FP32
I32 = inventory_gsq3.I32
Q3 = inventory_gsq3.Q3
Q4 = inventory_gsq3.Q4
Q5 = inventory_gsq3.Q5
Q6 = inventory_gsq3.Q6
W8 = inventory_gsq3.W8

FULL_ATTENTION_LAYERS = inventory_gsq3.FULL_ATTENTION_LAYERS
GDN_LAYERS = inventory_gsq3.GDN_LAYERS
RESOURCE_SPECS = inventory_gsq3.RESOURCE_SPECS
DRAFT_HEAD_TENSOR_SPECS = inventory_gsq3.DRAFT_HEAD_TENSOR_SPECS
MTP_TENSOR_SPECS = inventory_gsq3.MTP_TENSOR_SPECS
VISION_TENSOR_SPECS = inventory_gsq3.VISION_TENSOR_SPECS
DFLASH2_TENSOR_SPECS = inventory_gsq3.DFLASH2_TENSOR_SPECS

# GGML type names exactly as gguf-py reports them for the referenced release.
BYTESHAPE_FORMAT: Mapping[str, str] = {
    "IQ3_XXS": Q3,
    "IQ2_XXS": Q3,
    "IQ4_XS": Q4,
    "Q5_K": Q5,
    "Q6_K": Q6,
    "Q8_0": W8,
}

_ENDPOINTS = ("text/token_embedding", "text/output_head")

object_source_tensors = route_map.object_source_tensors
ported_object_names = route_map.ported_object_names


def object_format(name: str, registered_format: str, source_types: Mapping[str, str]) -> str:
    """Resolve one artifact object's format from the byteshape allocation."""

    return route_map.object_format(name, registered_format, source_types, BYTESHAPE_FORMAT)


def build_text_core_specs(
    source_types: Mapping[str, str], *, uniform_body: bool = False
) -> tuple[TensorSpec, ...]:
    """Object formats for the text core; ``uniform_body`` forces the GSQ3 grid."""

    specs: list[TensorSpec] = []
    for spec in inventory_gsq3.TEXT_CORE_TENSOR_SPECS:
        if uniform_body and spec.name not in _ENDPOINTS and object_source_tensors(spec.name) is not None:
            numeric_format = Q3
        else:
            numeric_format = object_format(spec.name, spec.format, source_types)
        specs.append(replace(spec, format=numeric_format))
    return tuple(specs)


def build_base_tensor_specs(
    source_types: Mapping[str, str], *, uniform_body: bool = False
) -> tuple[TensorSpec, ...]:
    """Text core, draft head, MTP and Vision in artifact order."""

    return (
        build_text_core_specs(source_types, uniform_body=uniform_body)
        + DRAFT_HEAD_TENSOR_SPECS
        + MTP_TENSOR_SPECS
        + VISION_TENSOR_SPECS
    )


def build_tensor_specs(
    source_types: Mapping[str, str], *, uniform_body: bool = False
) -> tuple[TensorSpec, ...]:
    return (
        build_text_core_specs(source_types, uniform_body=uniform_body)
        + DRAFT_HEAD_TENSOR_SPECS
        + MTP_TENSOR_SPECS
        + VISION_TENSOR_SPECS
        + DFLASH2_TENSOR_SPECS
    )


def build_object_specs(
    source_types: Mapping[str, str], *, uniform_body: bool = False
) -> tuple[StoredObjectSpec, ...]:
    return RESOURCE_SPECS + build_tensor_specs(source_types, uniform_body=uniform_body)


__all__ = [
    "BF16",
    "BYTESHAPE_FORMAT",
    "DFLASH2_TENSOR_SPECS",
    "DRAFT_HEAD_TENSOR_SPECS",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "I32",
    "MODEL_ID",
    "MTP_TENSOR_SPECS",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "VISION_TENSOR_SPECS",
    "W8",
    "WEIGHTS_ID",
    "build_base_tensor_specs",
    "build_object_specs",
    "build_tensor_specs",
    "build_text_core_specs",
    "object_format",
    "object_source_tensors",
    "ported_object_names",
]
