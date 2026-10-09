from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from tools.convert.qwen3_6.common import conversion as family_conversion
from tools.convert.qwen3_8_27b import convert_byteshape
from tools.convert.qwen3_8_27b import inventory_byteshape as inventory
from tools.convert.qwen3_8_27b import inventory_gsq3 as gsq3_inventory
from tools.convert.qwen3_8_27b import inventory_gsqrco as gsqrco_inventory
from tools.convert.qwen3_8_27b.gsqrco_source import gguf_tensor_types

_REAL_GGUF = Path(
    os.environ.get(
        "NINFER_BYTESHAPE_GGUF",
        "/models/qwen3.8-27b-byteshape-iq3s/Qwen3.8-27B-IQ3_S-3.23bpw.gguf",
    )
)


def _synthetic_types() -> dict[str, str]:
    """Per-tensor byteshape-shaped allocation exercising every mapped type."""

    types: dict[str, str] = {}
    for layer in range(64):
        types[f"blk.{layer}.attn_q.weight"] = "IQ3_XXS"
        types[f"blk.{layer}.attn_k.weight"] = "IQ3_XXS"
        types[f"blk.{layer}.attn_v.weight"] = "IQ4_XS"
        types[f"blk.{layer}.attn_output.weight"] = "IQ3_XXS"
        types[f"blk.{layer}.attn_qkv.weight"] = "IQ2_XXS"
        types[f"blk.{layer}.attn_gate.weight"] = "IQ3_XXS"
        types[f"blk.{layer}.ssm_out.weight"] = "Q5_K"
        types[f"blk.{layer}.ffn_gate.weight"] = "IQ3_XXS"
        types[f"blk.{layer}.ffn_up.weight"] = "IQ3_XXS"
        types[f"blk.{layer}.ffn_down.weight"] = "Q6_K"
    types["token_embd.weight"] = "IQ4_XS"
    types["output.weight"] = "Q6_K"
    return types


def test_ggml_types_map_to_registered_formats() -> None:
    assert inventory.BYTESHAPE_FORMAT == {
        "IQ3_XXS": "Q3G128_F16S",
        "IQ2_XXS": "Q3G128_F16S",
        "IQ4_XS": "Q4G64_F16S",
        "Q5_K": "Q5G64_F16S",
        "Q6_K": "Q6G64_F16S",
        "Q8_0": "W8G32_F16S",
    }


def test_fused_objects_take_the_widest_contributor() -> None:
    types = _synthetic_types()
    specs = {spec.name: spec for spec in inventory.build_tensor_specs(types)}
    assert specs["text/layers/3/attention/query_key"].format == "Q4G64_F16S"
    assert specs["text/layers/3/attention/gate_value"].format == "Q5G64_F16S"
    assert specs["text/layers/3/attention/output"].format == "Q3G128_F16S"
    assert specs["text/layers/0/gdn/query_key"].format == "Q3G128_F16S"
    assert specs["text/layers/0/gdn/value_z"].format == "Q3G128_F16S"
    assert specs["text/layers/0/gdn/output"].format == "Q5G64_F16S"
    assert specs["text/layers/0/mlp/gate_up"].format == "Q3G128_F16S"
    assert specs["text/layers/0/mlp/down"].format == "Q5G64_F16S"


def test_two_bit_sources_stay_on_the_three_bit_grid() -> None:
    types = _synthetic_types()
    specs = {spec.name: spec for spec in inventory.build_tensor_specs(types)}
    assert specs["text/layers/0/gdn/query_key"].format == "Q3G128_F16S"
    assert specs["text/layers/0/gdn/value_z"].format == "Q3G128_F16S"


def test_endpoints_stay_on_the_registered_vocabulary_grid() -> None:
    types = _synthetic_types()
    specs = {spec.name: spec for spec in inventory.build_tensor_specs(types)}
    assert specs["text/token_embedding"].format == "Q4G64_F16S"
    assert specs["text/output_head"].format == "Q4G64_F16S"


def test_only_text_core_matrix_objects_are_ported() -> None:
    ported = inventory.ported_object_names()
    assert ported == gsqrco_inventory.ported_object_names()
    assert len(ported) == 322
    assert "text/layers/0/mlp/down" in ported
    assert "text/token_embedding" in ported
    assert "text/layers/0/input_norm" not in ported
    assert "text/draft_head" not in ported
    assert "mtp/layer/mlp/down" not in ported
    assert "vision/patch_embedding" not in ported
    assert "dflash2/feature_projection" not in ported
    assert inventory.object_source_tensors("vision/patch_embedding") is None


def test_non_ported_specs_are_byte_identical_to_the_gsq3_plan() -> None:
    types = _synthetic_types()
    gsq3 = {spec.name: spec for spec in gsq3_inventory.TENSOR_SPECS}
    for spec in inventory.build_tensor_specs(types):
        if spec.name in inventory.ported_object_names():
            continue
        assert spec == gsq3[spec.name]


def test_object_plan_keeps_the_registered_order_and_count() -> None:
    resources = {spec.name: b"\x00" * 64 for spec in inventory.RESOURCE_SPECS}
    objects = family_conversion.build_object_plan(
        inventory.build_object_specs(_synthetic_types()), resources
    ).objects
    reference = family_conversion.build_object_plan(
        gsq3_inventory.OBJECT_SPECS, resources
    ).objects
    assert [obj.name for obj in objects] == [obj.name for obj in reference]
    assert len(objects) == len(reference)

    tensor_bytes = lambda entries: sum(obj.bytes for obj in entries if obj.kind == "tensor")
    assert tensor_bytes(objects) > tensor_bytes(reference)


def test_allocation_digest_is_order_independent() -> None:
    types = {"b": "IQ3_XXS", "a": "Q5_K"}
    assert convert_byteshape.allocation_digest(types) == convert_byteshape.allocation_digest(
        {"a": "Q5_K", "b": "IQ3_XXS"}
    )
    assert convert_byteshape.allocation_digest(types) != convert_byteshape.allocation_digest(
        {"a": "Q4G64_F16S", "b": "IQ3_XXS"}
    )


@pytest.mark.skipif(not _REAL_GGUF.is_file(), reason=f"{_REAL_GGUF} is absent")
def test_real_allocation_matches_the_published_mix() -> None:
    types = gguf_tensor_types(_REAL_GGUF)
    specs = {spec.name: spec for spec in inventory.build_tensor_specs(types)}
    ported = inventory.ported_object_names()
    counts = Counter(spec.format for name, spec in specs.items() if name in ported)
    assert counts == {"Q3G128_F16S": 271, "Q4G64_F16S": 23, "Q5G64_F16S": 28}
    assert specs["text/layers/0/gdn/output"].format == "Q5G64_F16S"
    assert specs["text/layers/1/gdn/query_key"].format == "Q3G128_F16S"
    assert specs["text/layers/27/attention/output"].format == "Q5G64_F16S"
    assert specs["text/layers/63/mlp/down"].format == "Q5G64_F16S"
    assert specs["text/token_embedding"].format == "Q4G64_F16S"
    assert specs["text/output_head"].format == "Q4G64_F16S"
    for name in ported:
        for source in inventory.object_source_tensors(name) or ():
            assert source in types, source
