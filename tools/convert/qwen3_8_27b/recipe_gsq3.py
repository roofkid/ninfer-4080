"""Artifact-object recipes for the Qwen3.8-27B GSQ 3-bit artifact.

Two expression languages meet here:

* the family :mod:`tools.convert.qwen3_6.common.recipe` language for BF16/FP32
  source tensors (norms, convolution, GDN A/B projections, MTP, Vision);
* a row-selection language over ``pack-quantized`` matrices
  (:class:`PackedSource`, :class:`PackedRows`, :class:`PackedConcat`) that never
  decodes a code or a scale, so every fused parent stays verbatim.

The 3-bit body objects reuse the registered groupwise row orders: full-attention
``query_key=[query,key]`` / ``gate_value=[gate,value]``, GDN
``query_key=[query,key]`` / ``value_z=[value,z]``, and ``mlp/gate_up``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import recipe as qwen3_6_recipe

from . import gsq3_source, inventory_gsq3 as inventory


SOURCE_BITS = 3
SOURCE_GROUP = 128
VOCAB_BITS = 4
VOCAB_GROUP = 64
ATTENTION_HEADS = 24
ATTENTION_HEAD_ROWS = 512
ATTENTION_PART_ROWS = 256


@dataclass(frozen=True, slots=True)
class PackedSource:
    name: str
    shape: tuple[int, int]
    bits: int
    group_size: int


@dataclass(frozen=True, slots=True)
class PackedRows:
    source: "PackedExpression"
    indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PackedConcat:
    sources: tuple["PackedExpression", ...]


@dataclass(frozen=True, slots=True)
class PackedGatherRows:
    """Select source rows in a derived token-id order without decoding."""

    source: "PackedExpression"
    token_ids_object: str
    rows: int


PackedExpression = PackedSource | PackedRows | PackedConcat | PackedGatherRows
Expression = PackedExpression | family_recipe.Expression

@dataclass(frozen=True, slots=True)
class ObjectRecipe:
    object_name: str
    expression: Expression


def packed_shape(expression: PackedExpression) -> tuple[int, int]:
    if isinstance(expression, PackedSource):
        return expression.shape
    if isinstance(expression, PackedRows):
        rows, columns = packed_shape(expression.source)
        if not expression.indices:
            raise ValueError("packed row selection requires at least one row")
        if min(expression.indices) < 0 or max(expression.indices) >= rows:
            raise ValueError("packed row selection is outside the source matrix")
        return (len(expression.indices), columns)
    if isinstance(expression, PackedConcat):
        if not expression.sources:
            raise ValueError("packed concatenation requires at least one matrix")
        shapes = [packed_shape(part) for part in expression.sources]
        columns = shapes[0][1]
        if any(shape[1] != columns for shape in shapes):
            raise ValueError("packed concatenation requires one K")
        return (sum(shape[0] for shape in shapes), columns)
    if isinstance(expression, PackedGatherRows):
        rows, columns = packed_shape(expression.source)
        if expression.rows <= 0 or expression.rows > rows:
            raise ValueError("packed gather exceeds the source matrix")
        return (expression.rows, columns)
    raise TypeError(f"unknown packed expression {type(expression)!r}")


def packed_geometry(expression: PackedExpression) -> tuple[int, int]:
    if isinstance(expression, PackedSource):
        return expression.bits, expression.group_size
    if isinstance(expression, PackedRows):
        return packed_geometry(expression.source)
    if isinstance(expression, PackedConcat):
        geometries = {packed_geometry(part) for part in expression.sources}
        if len(geometries) != 1:
            raise ValueError("packed concatenation requires one code geometry")
        return next(iter(geometries))
    if isinstance(expression, PackedGatherRows):
        return packed_geometry(expression.source)
    raise TypeError(f"unknown packed expression {type(expression)!r}")

def packed_sources(expression: PackedExpression) -> tuple[PackedSource, ...]:
    if isinstance(expression, PackedSource):
        return (expression,)
    if isinstance(expression, PackedRows):
        return packed_sources(expression.source)
    if isinstance(expression, PackedConcat):
        return tuple(
            source for part in expression.sources for source in packed_sources(part)
        )
    if isinstance(expression, PackedGatherRows):
        return packed_sources(expression.source)
    raise TypeError(f"unknown packed expression {type(expression)!r}")


def expression_shape(expression: Expression) -> tuple[int, ...]:
    if isinstance(expression, (PackedSource, PackedRows, PackedConcat, PackedGatherRows)):
        return packed_shape(expression)
    return family_recipe.expression_shape(expression)


def _qproj_rows(gate: bool) -> tuple[int, ...]:
    begin = ATTENTION_PART_ROWS if gate else 0
    return tuple(
        head * ATTENTION_HEAD_ROWS + begin + offset
        for head in range(ATTENTION_HEADS)
        for offset in range(ATTENTION_PART_ROWS)
    )


def _attention_indices() -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _qproj_rows(False), _qproj_rows(True)


def _layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}."


def _source(prefix: str, name: str, shape: tuple[int, int], bits: int = SOURCE_BITS,
            group_size: int = SOURCE_GROUP) -> PackedSource:
    return PackedSource(prefix + name.removesuffix(".weight"), shape, bits, group_size)


def _build_text_core_recipes() -> tuple[ObjectRecipe, ...]:
    recipes: list[ObjectRecipe] = [
        ObjectRecipe(
            "text/token_embedding",
            PackedSource(
                "model.language_model.embed_tokens",
                (248320, 5120),
                VOCAB_BITS,
                VOCAB_GROUP,
            ),
        )
    ]

    query_rows, gate_rows = _attention_indices()
    for layer in range(64):
        prefix = _layer_prefix(layer)
        object_prefix = f"text/layers/{layer}/"
        recipes.append(
            ObjectRecipe(
                object_prefix + "input_norm",
                qwen3_6_recipe.RECIPES_BY_NAME[object_prefix + "input_norm"].expression,
            )
        )
        if layer in inventory.FULL_ATTENTION_LAYERS:
            q_proj = _source(prefix, "self_attn.q_proj.weight", (12288, 5120))
            k_proj = _source(prefix, "self_attn.k_proj.weight", (1024, 5120))
            v_proj = _source(prefix, "self_attn.v_proj.weight", (1024, 5120))
            recipes.extend(
                (
                    ObjectRecipe(
                        object_prefix + "attention/query_key",
                        PackedConcat((PackedRows(q_proj, query_rows), k_proj)),
                    ),
                    ObjectRecipe(
                        object_prefix + "attention/gate_value",
                        PackedConcat((PackedRows(q_proj, gate_rows), v_proj)),
                    ),
                    ObjectRecipe(
                        object_prefix + "attention/query_norm",
                        qwen3_6_recipe.RECIPES_BY_NAME[
                            object_prefix + "attention/query_norm"
                        ].expression,
                    ),
                    ObjectRecipe(
                        object_prefix + "attention/key_norm",
                        qwen3_6_recipe.RECIPES_BY_NAME[
                            object_prefix + "attention/key_norm"
                        ].expression,
                    ),
                    ObjectRecipe(
                        object_prefix + "attention/output",
                        _source(prefix, "self_attn.o_proj.weight", (5120, 6144)),
                    ),
                )
            )
        else:
            qkv = _source(prefix, "linear_attn.in_proj_qkv.weight", (10240, 5120))
            recipes.extend(
                (
                    ObjectRecipe(
                        object_prefix + "gdn/a_log",
                        qwen3_6_recipe.RECIPES_BY_NAME[object_prefix + "gdn/a_log"].expression,
                    ),
                    ObjectRecipe(
                        object_prefix + "gdn/dt_bias",
                        qwen3_6_recipe.RECIPES_BY_NAME[object_prefix + "gdn/dt_bias"].expression,
                    ),
                    ObjectRecipe(
                        object_prefix + "gdn/convolution",
                        qwen3_6_recipe.RECIPES_BY_NAME[
                            object_prefix + "gdn/convolution"
                        ].expression,
                    ),
                    ObjectRecipe(
                        object_prefix + "gdn/a_projection",
                        qwen3_6_recipe.RECIPES_BY_NAME[
                            object_prefix + "gdn/a_projection"
                        ].expression,
                    ),
                    ObjectRecipe(
                        object_prefix + "gdn/b_projection",
                        qwen3_6_recipe.RECIPES_BY_NAME[
                            object_prefix + "gdn/b_projection"
                        ].expression,
                    ),
                    ObjectRecipe(
                        object_prefix + "gdn/query_key",
                        PackedRows(qkv, tuple(range(4096))),
                    ),
                    ObjectRecipe(
                        object_prefix + "gdn/value_z",
                        PackedConcat(
                            (
                                PackedRows(qkv, tuple(range(4096, 10240))),
                                _source(prefix, "linear_attn.in_proj_z.weight", (6144, 5120)),
                            )
                        ),
                    ),
                    ObjectRecipe(
                        object_prefix + "gdn/norm",
                        qwen3_6_recipe.RECIPES_BY_NAME[object_prefix + "gdn/norm"].expression,
                    ),
                    ObjectRecipe(
                        object_prefix + "gdn/output",
                        _source(prefix, "linear_attn.out_proj.weight", (5120, 6144)),
                    ),
                )
            )
        recipes.extend(
            (
                ObjectRecipe(
                    object_prefix + "post_attention_norm",
                    qwen3_6_recipe.RECIPES_BY_NAME[
                        object_prefix + "post_attention_norm"
                    ].expression,
                ),
                ObjectRecipe(
                    object_prefix + "mlp/gate_up",
                    PackedConcat(
                        (
                            _source(prefix, "mlp.gate_proj.weight", (17408, 5120)),
                            _source(prefix, "mlp.up_proj.weight", (17408, 5120)),
                        )
                    ),
                ),
                ObjectRecipe(
                    object_prefix + "mlp/down",
                    _source(prefix, "mlp.down_proj.weight", (5120, 17408)),
                ),
            )
        )

    recipes.extend(
        (
            ObjectRecipe(
                "text/final_norm",
                qwen3_6_recipe.RECIPES_BY_NAME["text/final_norm"].expression,
            ),
            ObjectRecipe(
                "text/output_head",
                PackedSource("lm_head", (248320, 5120), VOCAB_BITS, VOCAB_GROUP),
            ),
        )
    )
    return tuple(recipes)


def _build_draft_head_recipes() -> tuple[ObjectRecipe, ...]:
    draft = qwen3_6_recipe.RECIPES_BY_NAME["text/draft_head_token_ids"]
    return (
        ObjectRecipe(
            "text/draft_head",
            PackedGatherRows(
                PackedSource("lm_head", (248320, 5120), VOCAB_BITS, VOCAB_GROUP),
                token_ids_object="text/draft_head_token_ids",
                rows=131072,
            ),
        ),
        ObjectRecipe("text/draft_head_token_ids", draft.expression),
    )


def _build_mtp_recipes() -> tuple[ObjectRecipe, ...]:
    return tuple(
        ObjectRecipe(spec.name, qwen3_6_recipe.RECIPES_BY_NAME[spec.name].expression)
        for spec in inventory.MTP_TENSOR_SPECS
    )


def _build_vision_recipes() -> tuple[ObjectRecipe, ...]:
    return tuple(
        ObjectRecipe(spec.name, qwen3_6_recipe.RECIPES_BY_NAME[spec.name].expression)
        for spec in inventory.VISION_TENSOR_SPECS
    )


RECIPE_SPECS: tuple[ObjectRecipe, ...] = (
    _build_text_core_recipes()
    + _build_draft_head_recipes()
    + _build_mtp_recipes()
    + _build_vision_recipes()
)
RECIPES_BY_NAME = {recipe.object_name: recipe for recipe in RECIPE_SPECS}


def validate_recipe_coverage() -> None:
    inventory_names = tuple(spec.name for spec in inventory.TENSOR_SPECS)
    recipe_names = tuple(recipe.object_name for recipe in RECIPE_SPECS)
    if recipe_names != inventory_names:
        raise ValueError("recipe order or coverage does not match the tensor inventory")
    if len(RECIPES_BY_NAME) != len(RECIPE_SPECS):
        raise ValueError("more than one recipe targets the same artifact object")
    specs = inventory.tensor_specs_by_name()
    for recipe in RECIPE_SPECS:
        spec = specs[recipe.object_name]
        shape = expression_shape(recipe.expression)
        if shape != spec.shape:
            raise ValueError(
                f"{recipe.object_name}: recipe shape {shape} != inventory {spec.shape}"
            )
        if isinstance(
            recipe.expression,
            (PackedSource, PackedRows, PackedConcat, PackedGatherRows),
        ):
            bits, group_size = packed_geometry(recipe.expression)
            numeric = get_format(spec.format)
            if not isinstance(numeric, QuantFormat) or (
                numeric.bits,
                numeric.group_size,
            ) != (bits, group_size):
                raise ValueError(
                    f"{recipe.object_name}: packed geometry {(bits, group_size)} "
                    f"does not match artifact format {spec.format}"
                )


def preflight_inventory() -> None:
    validate_recipe_coverage()


def materialize_packed(
    expression: PackedExpression,
    reader: ShardReader,
    derived_tensors: Mapping[str, torch.Tensor] | None = None,
) -> gsq3_source.GsqMatrix:
    if isinstance(expression, PackedSource):
        matrix = gsq3_source.read_matrix(
            reader, expression.name, bits=expression.bits, group_size=expression.group_size
        )
        if (matrix.rows, matrix.columns) != expression.shape:
            raise ValueError(
                f"{expression.name}: source shape {(matrix.rows, matrix.columns)} "
                f"!= declared {expression.shape}"
            )
        return matrix
    if isinstance(expression, PackedRows):
        return materialize_packed(expression.source, reader, derived_tensors).select_rows(
            expression.indices
        )
    if isinstance(expression, PackedConcat):
        return gsq3_source.concat_rows(
            materialize_packed(part, reader, derived_tensors) for part in expression.sources
        )
    if isinstance(expression, PackedGatherRows):
        if derived_tensors is None or expression.token_ids_object not in derived_tensors:
            raise ValueError("packed gather requires the derived token ids")
        token_ids = derived_tensors[expression.token_ids_object].to(torch.long)
        selected = materialize_packed(expression.source, reader, derived_tensors).select_rows(
            token_ids
        )
        if selected.rows != expression.rows:
            raise ValueError("packed gather returned an unexpected row count")
        return selected
    raise TypeError(f"unknown packed expression {type(expression)!r}")


def packed_requirements() -> dict[str, PackedSource]:
    requirements: dict[str, PackedSource] = {}
    for recipe in RECIPE_SPECS:
        if not isinstance(recipe.expression, (PackedSource, PackedRows, PackedConcat)):
            continue
        for source in packed_sources(recipe.expression):
            previous = requirements.setdefault(source.name, source)
            if previous != source:
                raise ValueError(f"inconsistent packed declaration for {source.name}")
    return requirements


def validate_packed_sources(reader: ShardReader, requirements: Mapping[str, PackedSource]) -> int:
    for name, source in requirements.items():
        metadata = gsq3_source.read_metadata(
            reader, name, bits=source.bits, group_size=source.group_size
        )
        if (metadata.rows, metadata.columns) != source.shape:
            raise ValueError(
                f"{name}: source shape {(metadata.rows, metadata.columns)} != declared "
                f"{source.shape}"
            )
    return len(requirements)


def direct_recipes() -> tuple[ObjectRecipe, ...]:
    return tuple(
        recipe
        for recipe in RECIPE_SPECS
        if not isinstance(
            recipe.expression,
            (PackedSource, PackedRows, PackedConcat, PackedGatherRows),
        )
    )


def mtp_recipes() -> tuple[ObjectRecipe, ...]:
    selected: list[ObjectRecipe] = []
    for recipe in direct_recipes():
        names = {
            requirement.name
            for requirement in family_recipe.expression_sources(recipe.expression)
        }
        if names and all(name.startswith("mtp.") for name in names):
            selected.append(recipe)
    return tuple(selected)


def other_direct_recipes() -> tuple[ObjectRecipe, ...]:
    mtp = set(mtp_recipes())
    return tuple(recipe for recipe in direct_recipes() if recipe not in mtp)


def _as_family_recipes(
    recipes: Sequence[ObjectRecipe],
) -> tuple[family_recipe.TensorRecipe, ...]:
    return tuple(
        family_recipe.TensorRecipe(recipe.object_name, recipe.expression)
        for recipe in recipes
    )


def preflight_readers(
    gsq_reader: ShardReader,
    official_reader: ShardReader,
) -> tuple[int, family_recipe.SourcePreflight, family_recipe.SourcePreflight]:
    """Validate every declared source against its owning checkpoint reader."""

    requirements = packed_requirements()
    packed_count = validate_packed_sources(gsq_reader, requirements)
    mtp = family_recipe.preflight_source_reader(
        official_reader, _as_family_recipes(mtp_recipes())
    )
    direct = family_recipe.preflight_source_reader(
        gsq_reader, _as_family_recipes(other_direct_recipes())
    )
    return packed_count, mtp, direct

__all__ = [
    "ObjectRecipe",
    "PackedConcat",
    "PackedExpression",
    "PackedGatherRows",
    "PackedRows",
    "PackedSource",
    "RECIPES_BY_NAME",
    "RECIPE_SPECS",
    "direct_recipes",
    "expression_shape",
    "materialize_packed",
    "mtp_recipes",
    "other_direct_recipes",
    "packed_geometry",
    "packed_requirements",
    "packed_shape",
    "packed_sources",
    "preflight_inventory",
    "preflight_readers",
    "validate_packed_sources",
    "validate_recipe_coverage",
]
