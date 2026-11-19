"""Hugging Face source recipes for the SenseNova-Loop-9B artifact."""

from __future__ import annotations

from pathlib import Path

from tools.convert.qwen3_6.common.recipe import (
    SOURCE_DTYPE,
    Cast,
    Concat,
    Reshape,
    Slice,
    SourcePreflight,
    SourceTensor,
    TensorRecipe,
    Transpose,
    attention_qproj_part,
    build_vision_recipes,
    expression_shape,
    materialize_recipe,
    preflight_source_reader,
    preflight_sources as _preflight_recipe_sources,
    source,
    source_requirements,
    validate_recipe_coverage as _validate_recipe_coverage,
)
from tools.convert.common.safetensors import ShardReader

from . import inventory


_sources = source_requirements
_source = source


def _f32_source(name: str, shape: tuple[int, ...]) -> SourceTensor:
    return SourceTensor(name=name, shape=shape, dtype="F32")


def _attention_qproj_part(source_name: str, gate: bool):
    return attention_qproj_part(
        source_name,
        gate,
        num_heads=inventory.QUERY_HEADS,
        hidden_size=inventory.HIDDEN,
    )


def _build_text_recipes() -> tuple[TensorRecipe, ...]:
    recipes: list[TensorRecipe] = [
        TensorRecipe(
            "text/token_embedding",
            _source("model.language_model.embed_tokens.weight", (248320, inventory.HIDDEN)),
        )
    ]
    for layer in range(inventory.PHYSICAL_LAYERS):
        source_prefix = f"model.language_model.layers.{layer}."
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            TensorRecipe(
                object_prefix + "input_norm",
                _source(source_prefix + "input_layernorm.weight", (inventory.HIDDEN,)),
            )
        )
        if layer in inventory.FULL_ATTENTION_LAYERS:
            q_proj = source_prefix + "self_attn.q_proj.weight"
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "attention/query_key",
                        Concat(
                            (
                                _attention_qproj_part(q_proj, gate=False),
                                _source(source_prefix + "self_attn.k_proj.weight",
                                        (inventory.KV_SIZE, inventory.HIDDEN)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/gate_value",
                        Concat(
                            (
                                _attention_qproj_part(q_proj, gate=True),
                                _source(source_prefix + "self_attn.v_proj.weight",
                                        (inventory.KV_SIZE, inventory.HIDDEN)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/query_norm",
                        _source(source_prefix + "self_attn.q_norm.weight",
                                (inventory.HEAD_DIM,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/key_norm",
                        _source(source_prefix + "self_attn.k_norm.weight",
                                (inventory.HEAD_DIM,)),
                    ),
                    TensorRecipe(
                        object_prefix + "attention/output",
                        _source(source_prefix + "self_attn.o_proj.weight",
                                (inventory.HIDDEN, inventory.QUERY_SIZE)),
                    ),
                )
            )
        else:
            qkv_source = _source(source_prefix + "linear_attn.in_proj_qkv.weight",
                                 (2 * inventory.KEY_DIM + inventory.VALUE_DIM,
                                  inventory.HIDDEN))
            convolution = _source(source_prefix + "linear_attn.conv1d.weight",
                                  (inventory.CONV_CHANNELS, 1, 4))
            recipes.extend(
                (
                    TensorRecipe(
                        object_prefix + "gdn/a_log",
                        Cast(_f32_source(source_prefix + "linear_attn.A_log",
                                         (inventory.GDN_VALUE_HEADS,)), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/dt_bias",
                        Cast(_f32_source(source_prefix + "linear_attn.dt_bias",
                                         (inventory.GDN_VALUE_HEADS,)), inventory.FP32),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/convolution",
                        Transpose(
                            Reshape(Slice(convolution, 1, 0, 1), (inventory.CONV_CHANNELS, 4)),
                            (1, 0),
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/a_projection",
                        _source(source_prefix + "linear_attn.in_proj_a.weight",
                                (inventory.GDN_VALUE_HEADS, inventory.HIDDEN)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/b_projection",
                        _source(source_prefix + "linear_attn.in_proj_b.weight",
                                (inventory.GDN_VALUE_HEADS, inventory.HIDDEN)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/query_key",
                        Slice(qkv_source, 0, 0, 2 * inventory.KEY_DIM),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/value_z",
                        Concat(
                            (
                                Slice(qkv_source, 0, 2 * inventory.KEY_DIM,
                                      2 * inventory.KEY_DIM + inventory.VALUE_DIM),
                                _source(source_prefix + "linear_attn.in_proj_z.weight",
                                        (inventory.VALUE_DIM, inventory.HIDDEN)),
                            ),
                            0,
                        ),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/norm",
                        _source(source_prefix + "linear_attn.norm.weight",
                                (inventory.GDN_KEY_HEAD_DIM,)),
                    ),
                    TensorRecipe(
                        object_prefix + "gdn/output",
                        _source(source_prefix + "linear_attn.out_proj.weight",
                                (inventory.HIDDEN, inventory.VALUE_DIM)),
                    ),
                )
            )
        recipes.extend(
            (
                TensorRecipe(
                    object_prefix + "post_attention_norm",
                    _source(source_prefix + "post_attention_layernorm.weight",
                            (inventory.HIDDEN,)),
                ),
                TensorRecipe(
                    object_prefix + "mlp/gate_up",
                    Concat(
                        (
                            _source(source_prefix + "mlp.gate_proj.weight",
                                    (inventory.INTERMEDIATE, inventory.HIDDEN)),
                            _source(source_prefix + "mlp.up_proj.weight",
                                    (inventory.INTERMEDIATE, inventory.HIDDEN)),
                        ),
                        0,
                    ),
                ),
                TensorRecipe(
                    object_prefix + "mlp/down",
                    _source(source_prefix + "mlp.down_proj.weight",
                            (inventory.HIDDEN, inventory.INTERMEDIATE)),
                ),
            )
        )
        if layer in inventory.MHC_LAYERS:
            for router in ("attn", "ffn"):
                recipes.extend(
                    (
                        TensorRecipe(
                            object_prefix + f"mhc/{router}/weight",
                            _f32_source(source_prefix + f"mhc_{router}.weight",
                                        (8, 2 * inventory.HIDDEN)),
                        ),
                        TensorRecipe(
                            object_prefix + f"mhc/{router}/bias",
                            _f32_source(source_prefix + f"mhc_{router}.bias", (8,)),
                        ),
                    )
                )
    recipes.extend(
        (
            TensorRecipe(
                "text/final_norm",
                _source("model.language_model.norm.weight", (inventory.HIDDEN,)),
            ),
            TensorRecipe(
                "text/output_head",
                _source("lm_head.weight", (248320, inventory.HIDDEN)),
            ),
        )
    )
    return tuple(recipes)


def _build_mtp_recipes() -> tuple[TensorRecipe, ...]:
    q_proj = "mtp.layers.0.self_attn.q_proj.weight"
    return (
        TensorRecipe(
            "mtp/input_projection",
            _source("mtp.fc.weight", (inventory.HIDDEN, 2 * inventory.HIDDEN)),
        ),
        TensorRecipe(
            "mtp/embedding_norm",
            _source("mtp.pre_fc_norm_embedding.weight", (inventory.HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/hidden_norm",
            _source("mtp.pre_fc_norm_hidden.weight", (inventory.HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/layer/input_norm",
            _source("mtp.layers.0.input_layernorm.weight", (inventory.HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_key_gate_value",
            Concat(
                (
                    _attention_qproj_part(q_proj, gate=False),
                    _source("mtp.layers.0.self_attn.k_proj.weight",
                            (inventory.KV_SIZE, inventory.HIDDEN)),
                    _attention_qproj_part(q_proj, gate=True),
                    _source("mtp.layers.0.self_attn.v_proj.weight",
                            (inventory.KV_SIZE, inventory.HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/attention/query_norm",
            _source("mtp.layers.0.self_attn.q_norm.weight", (inventory.HEAD_DIM,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/key_norm",
            _source("mtp.layers.0.self_attn.k_norm.weight", (inventory.HEAD_DIM,)),
        ),
        TensorRecipe(
            "mtp/layer/attention/output",
            _source("mtp.layers.0.self_attn.o_proj.weight",
                    (inventory.HIDDEN, inventory.QUERY_SIZE)),
        ),
        TensorRecipe(
            "mtp/layer/post_attention_norm",
            _source("mtp.layers.0.post_attention_layernorm.weight", (inventory.HIDDEN,)),
        ),
        TensorRecipe(
            "mtp/layer/mlp/gate_up",
            Concat(
                (
                    _source("mtp.layers.0.mlp.gate_proj.weight",
                            (inventory.INTERMEDIATE, inventory.HIDDEN)),
                    _source("mtp.layers.0.mlp.up_proj.weight",
                            (inventory.INTERMEDIATE, inventory.HIDDEN)),
                ),
                0,
            ),
        ),
        TensorRecipe(
            "mtp/layer/mlp/down",
            _source("mtp.layers.0.mlp.down_proj.weight",
                    (inventory.HIDDEN, inventory.INTERMEDIATE)),
        ),
        TensorRecipe(
            "mtp/final_norm",
            _source("mtp.norm.weight", (inventory.HIDDEN,)),
        ),
    )


VISION_RECIPES = build_vision_recipes(inventory.HIDDEN)
TEXT_RECIPES = _build_text_recipes()
MTP_RECIPES = _build_mtp_recipes()
MAIN_RECIPES = TEXT_RECIPES + VISION_RECIPES
ALL_RECIPES = MAIN_RECIPES + MTP_RECIPES
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in ALL_RECIPES}


def validate_recipe_coverage() -> None:
    _validate_recipe_coverage(
        MAIN_RECIPES, inventory.TEXT_CORE_TENSOR_SPECS + inventory.VISION_TENSOR_SPECS
    )
    _validate_recipe_coverage(MTP_RECIPES, inventory.MTP_TENSOR_SPECS)
    if len(RECIPES_BY_NAME) != len(ALL_RECIPES):
        raise ValueError("more than one recipe targets the same artifact object")


def preflight_main(model_dir) -> SourcePreflight:
    return _preflight_recipe_sources(model_dir, MAIN_RECIPES)


def preflight_mtp(model_dir) -> SourcePreflight:
    with ShardReader.from_file(Path(model_dir) / "mtp.safetensors") as reader:
        return preflight_source_reader(reader, MTP_RECIPES)


def source_requirements_for(recipes):
    return _sources(recipes)
