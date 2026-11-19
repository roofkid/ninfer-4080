from __future__ import annotations

from tools.convert.sensenova_loop_9b import inventory


def test_registered_physical_topology() -> None:
    assert inventory.PHYSICAL_LAYERS == 32
    assert inventory.EXECUTION_SLOTS == 56
    assert inventory.FULL_ATTENTION_LAYERS == (3, 7, 11, 15, 19, 23, 27, 31)
    assert len(inventory.GDN_LAYERS) == 24
    assert inventory.MHC_LAYERS == tuple(range(4, 28))


def test_inventory_is_complete_and_unique() -> None:
    names = [spec.name for spec in inventory.OBJECT_SPECS]
    assert len(names) == len(set(names))
    assert len(inventory.RESOURCE_SPECS) == 6
    assert len(inventory.MTP_TENSOR_SPECS) == 12
    assert len(inventory.VISION_TENSOR_SPECS) == 333
    assert len(inventory.TENSOR_SPECS) == 828
    assert len(inventory.OBJECT_SPECS) == 834


def test_numeric_assignment() -> None:
    specs = {spec.name: spec for spec in inventory.TENSOR_SPECS}
    assert specs["text/token_embedding"].format == inventory.Q6
    assert specs["text/output_head"].format == inventory.Q6
    assert specs["text/layers/3/attention/query_key"].format == inventory.Q4
    assert specs["text/layers/3/attention/gate_value"].format == inventory.Q5
    assert specs["text/layers/0/gdn/query_key"].shape == (4096, 4096)
    assert specs["text/layers/0/gdn/value_z"].shape == (8192, 4096)
    assert specs["text/layers/4/mhc/attn/weight"].format == inventory.FP32
    assert specs["text/layers/4/mhc/attn/weight"].shape == (8, 8192)
    assert specs["mtp/input_projection"].format == inventory.W8
    assert specs["vision/patch_embedding"].format == inventory.Q6
    assert specs["vision/merger/fc2"].format == inventory.W8
    assert specs["vision/merger/fc2"].shape == (4096, 4608)
