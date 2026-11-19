"""Persistent-object contract for the SenseNova-Loop-9B groupwise-int artifact."""

from __future__ import annotations

from tools.convert.qwen3_6.common.inventory import (
    BF16,
    CONTIGUOUS_LAYOUT,
    DIRECT_FORMATS,
    FORMAT_NAMES,
    FP32,
    LAYOUT_NAMES,
    Q4,
    Q5,
    Q6,
    RESOURCE_ENCODING,
    RESOURCE_SPECS,
    ROW_SPLIT_LAYOUT,
    ResourceSpec,
    StoredObjectSpec,
    TensorSpec,
    VISION_LAYERS,
    W8,
    build_vision_specs,
    tensor_spec,
)


MODEL_ID = "sensenova-loop-9b"
WEIGHTS_ID = "groupwise-int"
TARGET_KEY = "sensenova_loop_9b"

PHYSICAL_LAYERS = 32
LOOP_START = 4
LOOP_END = 28
EXECUTION_SLOTS = 56
FULL_ATTENTION_LAYERS = tuple(layer for layer in range(PHYSICAL_LAYERS) if (layer + 1) % 4 == 0)
GDN_LAYERS = tuple(layer for layer in range(PHYSICAL_LAYERS) if layer not in FULL_ATTENTION_LAYERS)
MHC_LAYERS = tuple(range(LOOP_START, LOOP_END))

HIDDEN = 4096
INTERMEDIATE = 12288
QUERY_HEADS = 16
KV_HEADS = 4
HEAD_DIM = 256
GDN_KEY_HEADS = 16
GDN_KEY_HEAD_DIM = 128
GDN_VALUE_HEADS = 32
GDN_VALUE_HEAD_DIM = 128
QUERY_SIZE = QUERY_HEADS * HEAD_DIM
KV_SIZE = KV_HEADS * HEAD_DIM
KEY_DIM = GDN_KEY_HEADS * GDN_KEY_HEAD_DIM
VALUE_DIM = GDN_VALUE_HEADS * GDN_VALUE_HEAD_DIM
CONV_CHANNELS = 2 * KEY_DIM + VALUE_DIM


_tensor = tensor_spec


def _build_text_core_specs() -> tuple[TensorSpec, ...]:
    specs: list[TensorSpec] = [
        _tensor("text/token_embedding", (248320, HIDDEN), Q6),
    ]
    for layer in range(PHYSICAL_LAYERS):
        prefix = f"text/layers/{layer}/"
        specs.append(_tensor(prefix + "input_norm", (HIDDEN,), BF16))
        if layer in FULL_ATTENTION_LAYERS:
            specs.extend(
                (
                    _tensor(prefix + "attention/query_key",
                            (QUERY_SIZE + KV_SIZE, HIDDEN), Q4),
                    _tensor(prefix + "attention/gate_value",
                            (QUERY_SIZE + KV_SIZE, HIDDEN), Q5),
                    _tensor(prefix + "attention/query_norm", (HEAD_DIM,), BF16),
                    _tensor(prefix + "attention/key_norm", (HEAD_DIM,), BF16),
                    _tensor(prefix + "attention/output", (HIDDEN, QUERY_SIZE), Q5),
                )
            )
        else:
            specs.extend(
                (
                    _tensor(prefix + "gdn/a_log", (GDN_VALUE_HEADS,), FP32),
                    _tensor(prefix + "gdn/dt_bias", (GDN_VALUE_HEADS,), FP32),
                    _tensor(prefix + "gdn/convolution", (4, CONV_CHANNELS), BF16),
                    _tensor(prefix + "gdn/a_projection", (GDN_VALUE_HEADS, HIDDEN), BF16),
                    _tensor(prefix + "gdn/b_projection", (GDN_VALUE_HEADS, HIDDEN), BF16),
                    _tensor(prefix + "gdn/query_key", (2 * KEY_DIM, HIDDEN), Q4),
                    _tensor(prefix + "gdn/value_z", (2 * VALUE_DIM, HIDDEN), Q5),
                    _tensor(prefix + "gdn/norm", (GDN_KEY_HEAD_DIM,), BF16),
                    _tensor(prefix + "gdn/output", (HIDDEN, VALUE_DIM), Q5),
                )
            )
        specs.extend(
            (
                _tensor(prefix + "post_attention_norm", (HIDDEN,), BF16),
                _tensor(prefix + "mlp/gate_up", (2 * INTERMEDIATE, HIDDEN), Q4),
                _tensor(prefix + "mlp/down", (HIDDEN, INTERMEDIATE), Q5),
            )
        )
        if layer in MHC_LAYERS:
            for router in ("attn", "ffn"):
                specs.extend(
                    (
                        _tensor(prefix + f"mhc/{router}/weight", (8, 2 * HIDDEN), FP32),
                        _tensor(prefix + f"mhc/{router}/bias", (8,), FP32),
                    )
                )
    specs.extend(
        (
            _tensor("text/final_norm", (HIDDEN,), BF16),
            _tensor("text/output_head", (248320, HIDDEN), Q6),
        )
    )
    return tuple(specs)


def _build_mtp_specs() -> tuple[TensorSpec, ...]:
    return (
        _tensor("mtp/input_projection", (HIDDEN, 2 * HIDDEN), W8),
        _tensor("mtp/embedding_norm", (HIDDEN,), BF16),
        _tensor("mtp/hidden_norm", (HIDDEN,), BF16),
        _tensor("mtp/layer/input_norm", (HIDDEN,), BF16),
        _tensor("mtp/layer/attention/query_key_gate_value",
                (2 * QUERY_SIZE + 2 * KV_SIZE, HIDDEN), W8),
        _tensor("mtp/layer/attention/query_norm", (HEAD_DIM,), BF16),
        _tensor("mtp/layer/attention/key_norm", (HEAD_DIM,), BF16),
        _tensor("mtp/layer/attention/output", (HIDDEN, QUERY_SIZE), W8),
        _tensor("mtp/layer/post_attention_norm", (HIDDEN,), BF16),
        _tensor("mtp/layer/mlp/gate_up", (2 * INTERMEDIATE, HIDDEN), W8),
        _tensor("mtp/layer/mlp/down", (HIDDEN, INTERMEDIATE), W8),
        _tensor("mtp/final_norm", (HIDDEN,), BF16),
    )


def _build_vision_specs() -> tuple[TensorSpec, ...]:
    return build_vision_specs(HIDDEN)


TEXT_CORE_TENSOR_SPECS = _build_text_core_specs()
MTP_TENSOR_SPECS = _build_mtp_specs()
VISION_TENSOR_SPECS = _build_vision_specs()
TENSOR_SPECS = TEXT_CORE_TENSOR_SPECS + MTP_TENSOR_SPECS + VISION_TENSOR_SPECS
MTP_OBJECT_NAMES = frozenset(spec.name for spec in MTP_TENSOR_SPECS)
OBJECT_SPECS: tuple[StoredObjectSpec, ...] = RESOURCE_SPECS + TENSOR_SPECS

FORMAT_COUNTS = {
    numeric_format: sum(spec.format == numeric_format for spec in TENSOR_SPECS)
    for numeric_format in FORMAT_NAMES
}
LAYOUT_COUNTS = {
    layout: sum(spec.layout == layout for spec in TENSOR_SPECS) for layout in LAYOUT_NAMES
}
