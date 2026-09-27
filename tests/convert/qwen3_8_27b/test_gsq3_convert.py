from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.artifact.container import Artifact
from tools.convert.qwen3_8_27b import convert_gsq3 as convert
from tools.convert.qwen3_8_27b import inventory_gsq3 as inventory
from tools.convert.qwen3_8_27b import recipe_gsq3 as recipe


def _tensors() -> dict[str, inventory.TensorSpec]:
    return {spec.name: spec for spec in inventory.TENSOR_SPECS}


def test_format_assignment_matches_the_source_scheme() -> None:
    tensors = _tensors()
    assert tensors["text/token_embedding"].format == "Q4G64_F16S"
    assert tensors["text/output_head"].format == "Q4G64_F16S"
    assert tensors["text/draft_head"].format == "Q4G64_F16S"
    assert tensors["text/draft_head_token_ids"].format == "I32"
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        for suffix in ("mlp/gate_up", "mlp/down"):
            assert tensors[prefix + suffix].format == "Q3G128_F16S"
        if layer in inventory.FULL_ATTENTION_LAYERS:
            assert tensors[prefix + "attention/query_key"].format == "Q3G128_F16S"
            assert tensors[prefix + "attention/gate_value"].format == "Q3G128_F16S"
            assert tensors[prefix + "attention/output"].format == "Q3G128_F16S"
        else:
            assert tensors[prefix + "gdn/query_key"].format == "Q3G128_F16S"
            assert tensors[prefix + "gdn/value_z"].format == "Q3G128_F16S"
            assert tensors[prefix + "gdn/output"].format == "Q3G128_F16S"
        assert tensors[prefix + "input_norm"].format == "BF16"
        assert tensors[prefix + "post_attention_norm"].format == "BF16"
    assert tensors["mtp/layer/mlp/gate_up"].format == "W8G32_F16S"
    assert tensors["vision/patch_embedding"].format == "Q6G64_F16S"
    assert all(not spec.name.startswith("dflash2/") for spec in inventory.TENSOR_SPECS)


def test_inventory_and_recipe_coverage_are_the_registered_shape() -> None:
    assert len(inventory.RESOURCE_SPECS) == 6
    assert len(inventory.TEXT_CORE_TENSOR_SPECS) == 771
    assert len(inventory.DRAFT_HEAD_TENSOR_SPECS) == 2
    assert len(inventory.MTP_TENSOR_SPECS) == 12
    assert len(inventory.VISION_TENSOR_SPECS) == 333
    assert len(inventory.TENSOR_SPECS) == 1118
    assert len(inventory.OBJECT_SPECS) == 1124
    recipe.validate_recipe_coverage()
    assert len(recipe.RECIPE_SPECS) == len(inventory.TENSOR_SPECS)
    assert len(recipe.packed_requirements()) == 402


def test_packed_geometry_matches_the_registered_formats() -> None:
    for name, source in recipe.packed_requirements().items():
        if name in ("lm_head", "model.language_model.embed_tokens"):
            assert (source.bits, source.group_size) == (4, 64)
        else:
            assert (source.bits, source.group_size) == (3, 128)


def test_config_validation_rejects_tampered_quantization_scheme() -> None:
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "language_model_only": False,
        "tie_word_embeddings": False,
        "vision_start_token_id": 248053,
        "vision_end_token_id": 248054,
        "image_token_id": 248056,
        "video_token_id": 248057,
        "text_config": {
            "num_hidden_layers": 64,
            "full_attention_interval": 4,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "mamba_ssm_dtype": "float32",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
            "tie_word_embeddings": False,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "layer_types": [
                "full_attention" if layer % 4 == 3 else "linear_attention"
                for layer in range(64)
            ],
            "rope_parameters": {
                "rope_theta": 10000000,
                "mrope_section": [11, 11, 10],
            },
        },
        "vision_config": {
            "depth": 27,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "out_hidden_size": 5120,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 2304,
        },
        "quantization_config": {
            "format": "pack-quantized",
            "quant_method": "compressed-tensors",
            "ignore": list(convert._IGNORE),
            "config_groups": {
                "group_0": {
                    "targets": list(convert._GROUP_0_TARGETS),
                    "input_activations": None,
                    "output_activations": None,
                    "weights": {
                        "num_bits": 3,
                        "group_size": 128,
                        "symmetric": True,
                        "strategy": "group",
                        "type": "int",
                    },
                },
                "group_1": {
                    "targets": [r"re:.*embed_tokens.*", r"re:.*lm_head$"],
                    "input_activations": None,
                    "output_activations": None,
                    "weights": {
                        "num_bits": 4,
                        "group_size": 64,
                        "symmetric": True,
                        "strategy": "group",
                        "type": "int",
                    },
                },
            },
        },
    }
    summary = convert.validate_config(config)
    assert summary["mtp_num_hidden_layers"] == 1

    tampered = dict(config)
    tampered["quantization_config"] = {
        **config["quantization_config"],
        "config_groups": {
            **config["quantization_config"]["config_groups"],
            "group_0": {
                **config["quantization_config"]["config_groups"]["group_0"],
                "weights": {
                    **config["quantization_config"]["config_groups"]["group_0"]["weights"],
                    "num_bits": 4,
                },
            },
        },
    }
    with pytest.raises(ValueError, match="num_bits"):
        convert.validate_config(tampered)


def test_object_plan_is_one_preplanned_directory() -> None:
    resources = {spec.name: b"x" for spec in inventory.RESOURCE_SPECS}
    plan = convert.build_object_plan(resources)
    assert tuple(spec.name for spec in plan.specs) == tuple(
        spec.name for spec in inventory.OBJECT_SPECS
    )
    assert tuple(obj.name for obj in plan.objects) == tuple(
        spec.name for spec in inventory.OBJECT_SPECS
    )


def _gsq_model_dir() -> Path:
    value = os.environ.get("NINFER_QWEN3_8_27B_GSQ3_MODEL")
    if not value:
        pytest.skip("NINFER_QWEN3_8_27B_GSQ3_MODEL is not set")
    return Path(value)


def test_real_source_preflight() -> None:
    gsq_dir = _gsq_model_dir()
    official = os.environ.get("NINFER_QWEN3_8_27B_OFFICIAL_MODEL")
    if not official:
        pytest.skip("NINFER_QWEN3_8_27B_OFFICIAL_MODEL is not set")
    preflight = convert.preflight_conversion(gsq_dir, Path(official))
    assert len(preflight.object_plan.objects) == 1124
    assert preflight.packed_source_count == 402


def test_real_artifact_structure() -> None:
    artifact_path = os.environ.get("NINFER_QWEN3_8_27B_GSQ3_ARTIFACT")
    if not artifact_path:
        pytest.skip("NINFER_QWEN3_8_27B_GSQ3_ARTIFACT is not set")
    from tools.convert.qwen3_8_27b.verify_gsq3 import validate_structure

    with Artifact.open(Path(artifact_path)) as artifact:
        summary = validate_structure(artifact)
    assert summary.objects == 1124
    assert summary.tensors == 1118
