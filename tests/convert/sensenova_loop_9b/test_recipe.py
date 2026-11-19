from __future__ import annotations

from tools.convert.sensenova_loop_9b import inventory, recipe


def test_recipe_coverage_and_shapes() -> None:
    recipe.validate_recipe_coverage()
    assert len(recipe.MAIN_RECIPES) == len(inventory.TEXT_CORE_TENSOR_SPECS) + len(
        inventory.VISION_TENSOR_SPECS
    )
    assert len(recipe.MTP_RECIPES) == len(inventory.MTP_TENSOR_SPECS)


def test_mhc_and_mtp_source_mapping() -> None:
    by_name = {entry.object_name: entry for entry in recipe.ALL_RECIPES}
    mhc = by_name["text/layers/4/mhc/attn/weight"]
    assert mhc.expression.name == "model.language_model.layers.4.mhc_attn.weight"
    mtp = by_name["mtp/layer/attention/query_key_gate_value"]
    assert recipe.expression_shape(mtp.expression) == (10240, 4096)
    assert by_name["mtp/final_norm"].expression.name == "mtp.norm.weight"
