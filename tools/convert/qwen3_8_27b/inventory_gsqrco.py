"""Persistent-object contract for the Qwen3.8-27B GSQ-RCO IQ3_S port.

The object plan mirrors :mod:`inventory_gsq3` object-for-object.  Only the
text-core matrix format follows the published RCO allocation of
``Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf``: IQ4_XS/Q4_K map to Q4G64_F16S,
IQ3_S/IQ3_XXS map to Q3G128_F16S, and the IQ2_*/Q2_K/IQ1_M tensors are promoted
to Q3G128_F16S because the artifact has no 2-bit family (the port is then never
less precise than the GGUF per source tensor).  Fused objects (``query_key``,
``gate_value``, ``value_z``, ``gate_up``) take the widest contributor, because
one artifact object carries one numeric format.

The vocabulary endpoints keep the registered Q4G64 grid (the embedding route has
no 3-bit table or gather path, and the GGUF's IQ2_S embedding is promoted, not
degraded).  Vision, DFlash2, the draft head, MTP and every resource are
identical to the GSQ3 contract and are copied from that artifact by the
converter.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from tools.convert.qwen3_6.common.inventory import StoredObjectSpec, TensorSpec

from . import inventory_gsq3

MODEL_ID = inventory_gsq3.MODEL_ID
WEIGHTS_ID = "gsqrco-iq3s"
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

# GGML type names exactly as gguf-py and the RCO allocation file report them.
GGUF_FORMAT: Mapping[str, str] = {
    "Q6_K": Q6,
    "Q4_K": Q4,
    "IQ4_XS": Q4,
    "IQ3_S": Q3,
    "IQ3_XXS": Q3,
    "IQ2_S": Q3,
    "IQ2_XS": Q3,
    "IQ2_XXS": Q3,
    "Q2_K": Q3,
    "IQ1_M": Q3,
}

_FORMAT_RANK = {Q3: 3, Q4: 4, Q5: 5, Q6: 6, W8: 8}

_ENDPOINTS = ("text/token_embedding", "text/output_head")


def object_source_tensors(name: str) -> tuple[str, ...] | None:
    """GGUF tensors that contribute rows to one ported artifact object.

    ``None`` marks an object that is copied from the source artifact instead of
    being encoded from the GGUF.
    """

    if name == "text/token_embedding":
        return ("token_embd.weight",)
    if name == "text/output_head":
        return ("output.weight",)
    parts = name.split("/")
    if len(parts) < 4 or parts[0] != "text" or parts[1] != "layers":
        return None
    layer = int(parts[2])
    suffix = "/".join(parts[3:])
    full_attention = layer in FULL_ATTENTION_LAYERS
    if suffix.startswith("attention/") and not full_attention:
        return None
    if suffix.startswith("gdn/") and full_attention:
        return None
    if suffix == "attention/query_key":
        return (f"blk.{layer}.attn_q.weight", f"blk.{layer}.attn_k.weight")
    if suffix == "attention/gate_value":
        return (f"blk.{layer}.attn_q.weight", f"blk.{layer}.attn_v.weight")
    if suffix == "attention/output":
        return (f"blk.{layer}.attn_output.weight",)
    if suffix == "gdn/query_key":
        return (f"blk.{layer}.attn_qkv.weight",)
    if suffix == "gdn/value_z":
        return (f"blk.{layer}.attn_qkv.weight", f"blk.{layer}.attn_gate.weight")
    if suffix == "gdn/output":
        return (f"blk.{layer}.ssm_out.weight",)
    if suffix == "mlp/gate_up":
        return (f"blk.{layer}.ffn_gate.weight", f"blk.{layer}.ffn_up.weight")
    if suffix == "mlp/down":
        return (f"blk.{layer}.ffn_down.weight",)
    return None


def _source_rank(source_types: Mapping[str, str], sources: tuple[str, ...]) -> int:
    """Highest registered-format rank among the GGUF sources of one object."""

    ranks: list[int] = []
    for source in sources:
        ggml_type = source_types.get(source)
        if ggml_type is None:
            raise KeyError(f"source tensor {source!r} is absent from the GGUF")
        try:
            ranks.append(_FORMAT_RANK[GGUF_FORMAT[ggml_type]])
        except KeyError as error:
            raise ValueError(f"{source}: unsupported GGML type {ggml_type!r}") from error
    return max(ranks)


def object_format(name: str, registered_format: str, source_types: Mapping[str, str]) -> str:
    """Resolve one artifact object's format from the GGUF per-tensor types."""

    if name in _ENDPOINTS:
        return Q4
    sources = object_source_tensors(name)
    if sources is None:
        return registered_format
    parts = name.split("/")
    layer = int(parts[2])
    suffix = "/".join(parts[3:])
    prefix = f"blk.{layer}."
    if layer in FULL_ATTENTION_LAYERS:
        if suffix in ("attention/query_key", "attention/gate_value"):
            rank = _source_rank(
                source_types,
                (prefix + "attn_q.weight", prefix + "attn_k.weight", prefix + "attn_v.weight"),
            )
            if rank >= 4:
                return Q4 if suffix == "attention/query_key" else Q5
            return Q3
        if suffix == "attention/output":
            rank = _source_rank(source_types, (prefix + "attn_output.weight",))
            return Q5 if rank >= 4 else Q3
    else:
        if suffix in ("gdn/query_key", "gdn/value_z"):
            rank = _source_rank(source_types, (prefix + "attn_qkv.weight",
                                                prefix + "attn_gate.weight"))
            if rank >= 4:
                return Q4 if suffix == "gdn/query_key" else Q5
            return Q3
        if suffix == "gdn/output":
            rank = _source_rank(source_types, (prefix + "ssm_out.weight",))
            return Q5 if rank >= 4 else Q3
    if suffix == "mlp/gate_up":
        rank = _source_rank(source_types, (prefix + "ffn_gate.weight", prefix + "ffn_up.weight"))
        return Q4 if rank >= 4 else Q3
    if suffix == "mlp/down":
        rank = _source_rank(source_types, (prefix + "ffn_down.weight",))
        return Q5 if rank >= 4 else Q3
    return registered_format


def build_text_core_specs(
    source_types: Mapping[str, str], *, uniform_body: bool = False
) -> tuple[TensorSpec, ...]:
    """Object formats for the text core; ``uniform_body`` forces the GSQ3 grid."""

    specs: list[TensorSpec] = []
    for spec in inventory_gsq3.TEXT_CORE_TENSOR_SPECS:
        if (
            uniform_body
            and spec.name not in _ENDPOINTS
            and object_source_tensors(spec.name) is not None
        ):
            numeric_format = Q3
        else:
            numeric_format = object_format(spec.name, spec.format, source_types)
        specs.append(replace(spec, format=numeric_format))
    return tuple(specs)


def build_tensor_specs(
    source_types: Mapping[str, str], *, uniform_body: bool = False
) -> tuple[TensorSpec, ...]:
    return (
        build_text_core_specs(source_types, uniform_body=uniform_body)
        + inventory_gsq3.DRAFT_HEAD_TENSOR_SPECS
        + inventory_gsq3.MTP_TENSOR_SPECS
        + inventory_gsq3.VISION_TENSOR_SPECS
        + inventory_gsq3.DFLASH2_TENSOR_SPECS
    )


def build_object_specs(
    source_types: Mapping[str, str], *, uniform_body: bool = False
) -> tuple[StoredObjectSpec, ...]:
    return RESOURCE_SPECS + build_tensor_specs(source_types, uniform_body=uniform_body)


def ported_object_names() -> frozenset[str]:
    """Text-core objects whose values are re-encoded from the GGUF."""

    names = set(_ENDPOINTS)
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        if layer in FULL_ATTENTION_LAYERS:
            suffixes = (
                "attention/query_key",
                "attention/gate_value",
                "attention/output",
                "mlp/gate_up",
                "mlp/down",
            )
        else:
            suffixes = (
                "gdn/query_key",
                "gdn/value_z",
                "gdn/output",
                "mlp/gate_up",
                "mlp/down",
            )
        names.update(prefix + suffix for suffix in suffixes)
    return frozenset(names)


__all__ = [
    "BF16",
    "FP32",
    "FULL_ATTENTION_LAYERS",
    "GDN_LAYERS",
    "GGUF_FORMAT",
    "I32",
    "MODEL_ID",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "RESOURCE_SPECS",
    "TARGET_KEY",
    "W8",
    "WEIGHTS_ID",
    "build_object_specs",
    "build_tensor_specs",
    "build_text_core_specs",
    "object_format",
    "object_source_tensors",
    "ported_object_names",
]
