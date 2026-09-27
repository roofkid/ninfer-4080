"""Independent verification for the Qwen3.8-27B GSQ 3-bit artifact.

The gate is stronger than the groupwise verify: every packed artifact plane is
rebuilt from the source shards with a verifier-local row plan and a verifier
local bit flip, then compared byte-for-byte, and every binary16 multiplier is
compared word-for-word against the bf16 source with the subnormal rounding
audit.  Direct and locally quantized objects (norms, convolution, Vision, MTP)
reuse the registered representative-row oracle.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Sequence
import torch

from tools.artifact.container import (
    Artifact,
    ArtifactIdentity,
    ResourceObject,
    TensorObject,
    object_alignment,
)
from tools.artifact.layouts import (
    align_up,
    decode_direct,
    decode_row_split_codes,
    encoded_size,
    row_split_geometry,
)
from tools.artifact.numeric import get_format
from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_6.common import recipe as family_recipe
from tools.convert.qwen3_6_27b import verify as family_verify

from . import convert_gsq3, gsq3_source, inventory_gsq3 as inventory
from . import recipe_gsq3 as recipe


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DIRECT_PROBE_OBJECTS = (
    "text/layers/0/input_norm",
    "text/layers/0/gdn/a_log",
    "text/layers/0/gdn/convolution",
)
QUANTIZED_PROBE_OBJECTS = (
    "vision/patch_embedding",
    "vision/layers/0/attention/qkv",
    "vision/merger/fc1",
    "mtp/layer/mlp/gate_up",
    "mtp/layer/attention/query_key_gate_value",
)

# Verifier-local flip of the sign bit of every source code.  The publisher
# stores ``value + 2**(bits-1)``; the artifact stores the two's complement code.
_SIGN_MASK_PERIOD = {3: bytes((0x24, 0x49, 0x92)), 4: bytes((0x88,))}


class VerificationError(ValueError):
    """The artifact does not match the registered GSQ3 target contract."""


@dataclass(frozen=True, slots=True)
class StructureSummary:
    objects: int
    tensors: int
    resources: int
    payload_bytes: int
    row_view_templates: int
    row_view_bindings: int
    alias_templates: int
    alias_bindings: int


@dataclass(frozen=True, slots=True)
class PackedSummary:
    objects: int
    rows: int
    groups: int
    base_bytes_equal: int
    scales_equal: int
    scales_rounded: int
    max_scale_error: float


@dataclass(frozen=True, slots=True)
class PayloadSummary:
    direct_probes: int
    quantized_probes: int
    quantized_rows: int
    quantized_groups: int
    packed: PackedSummary
    draft_rows: int
    resources: int
    processor_class: str
    generation_config_class: str


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    structure: StructureSummary
    payload: PayloadSummary


def _contract_error(message: str) -> None:
    raise VerificationError(message)


def _object_index(objects: Sequence[ResourceObject | TensorObject]):
    return {obj.name: obj for obj in objects}


def validate_logical_bindings(
    objects: Sequence[ResourceObject | TensorObject],
) -> tuple[int, int]:
    """Validate every fixed row view and alias against bound physical objects."""

    index = _object_index(objects)
    row_bindings = 0
    for view in inventory.LOGICAL_ROW_VIEW_SPECS:
        layers: tuple[int | None, ...]
        layers = (None,) if view.layers is None else view.layers
        for layer in layers:
            parent_name = (
                view.parent_pattern
                if layer is None
                else view.parent_pattern.format(l=layer)
            )
            parent = index.get(parent_name)
            if not isinstance(parent, TensorObject):
                _contract_error(f"logical view parent is missing: {parent_name}")
            if len(parent.shape) != 2:
                _contract_error(f"logical view parent is not a matrix: {parent_name}")
            if view.row_end > parent.shape[0]:
                _contract_error(f"logical view exceeds parent rows: {view.name_pattern}")
            if view.shape != (view.row_end - view.row_begin, parent.shape[1]):
                _contract_error(f"logical view shape is inconsistent: {view.name_pattern}")
            row_bindings += 1

    alias_bindings = 0
    for alias in inventory.ALIAS_SPECS:
        layers = (None,) if alias.layers is None else alias.layers
        for layer in layers:
            names = tuple(
                pattern if layer is None else pattern.format(l=layer)
                for pattern in alias.object_patterns
            )
            bound = [index.get(name) for name in names]
            if any(obj is None for obj in bound):
                _contract_error(f"logical alias has a missing object: {alias.role_pattern}")
            if alias.axis_order is not None:
                if len(bound) != 1 or not isinstance(bound[0], TensorObject):
                    _contract_error(f"axis alias does not bind one tensor: {alias.role_pattern}")
                source_shape = bound[0].shape
                if tuple(sorted(alias.axis_order)) != tuple(range(len(source_shape))):
                    _contract_error(f"axis alias is invalid: {alias.role_pattern}")
                target_shape = tuple(source_shape[axis] for axis in alias.axis_order)
                if target_shape != (10240, 4):
                    _contract_error(f"GDN convolution alias has shape {target_shape}")
            alias_bindings += 1

    return row_bindings, alias_bindings


def validate_structure(artifact: Artifact) -> StructureSummary:
    """Validate the complete directory without reading tensor payload values."""

    expected_identity = ArtifactIdentity(inventory.MODEL_ID, inventory.WEIGHTS_ID)
    if artifact.identity != expected_identity:
        _contract_error(
            f"artifact identity is {artifact.identity!r}, expected {expected_identity!r}"
        )
    if len(artifact.objects) != len(inventory.OBJECT_SPECS):
        _contract_error(
            f"artifact has {len(artifact.objects)} objects, expected "
            f"{len(inventory.OBJECT_SPECS)}"
        )

    cursor = 0
    tensor_count = 0
    resource_count = 0
    formats: Counter[str] = Counter()
    layouts: Counter[str] = Counter()
    for position, (actual, expected) in enumerate(
        zip(artifact.objects, inventory.OBJECT_SPECS)
    ):
        if actual.name != expected.name:
            _contract_error(
                f"object {position} is {actual.name!r}, expected {expected.name!r}"
            )
        expected_offset = align_up(cursor, object_alignment(actual))
        if actual.offset != expected_offset:
            _contract_error(
                f"{actual.name}: offset {actual.offset}, expected {expected_offset}"
            )

        if isinstance(expected, inventory.TensorSpec):
            if not isinstance(actual, TensorObject):
                _contract_error(f"{actual.name}: expected a tensor descriptor")
            signature = (actual.shape, actual.format, actual.layout)
            registered = (expected.shape, expected.format, expected.layout)
            if signature != registered:
                _contract_error(
                    f"{actual.name}: signature {signature} does not match {registered}"
                )
            required_bytes = encoded_size(actual.layout, actual.format, actual.shape)
            if actual.bytes != required_bytes:
                _contract_error(
                    f"{actual.name}: stores {actual.bytes} bytes, expected {required_bytes}"
                )
            tensor_count += 1
            formats[actual.format] += 1
            layouts[actual.layout] += 1
        else:
            if not isinstance(actual, ResourceObject):
                _contract_error(f"{actual.name}: expected a resource descriptor")
            if actual.encoding != expected.encoding:
                _contract_error(
                    f"{actual.name}: encoding {actual.encoding!r}, expected {expected.encoding!r}"
                )
            resource_count += 1

        cursor = actual.offset + actual.bytes

    if dict(formats) != inventory.FORMAT_COUNTS:
        _contract_error(f"numeric-format counts are {dict(formats)}")
    if dict(layouts) != inventory.LAYOUT_COUNTS:
        _contract_error(f"layout counts are {dict(layouts)}")

    payload_bytes = artifact.file_bytes - artifact.payload_offset
    if cursor != payload_bytes:
        _contract_error(f"payload ends at {cursor}, file contains {payload_bytes} bytes")

    row_bindings, alias_bindings = validate_logical_bindings(artifact.objects)
    return StructureSummary(
        objects=len(artifact.objects),
        tensors=tensor_count,
        resources=resource_count,
        payload_bytes=payload_bytes,
        row_view_templates=len(inventory.LOGICAL_ROW_VIEW_SPECS),
        row_view_bindings=row_bindings,
        alias_templates=len(inventory.ALIAS_SPECS),
        alias_bindings=alias_bindings,
    )


def _all_rows(prefix: str, rows: int) -> tuple[str, tuple[int, ...]]:
    return prefix, tuple(range(rows))


def _qproj_part(prefix: str, gate: bool) -> tuple[str, tuple[int, ...]]:
    # Independent inverse of the interleave: output row r belongs to head r//256
    # and is the (r % 256)-th row of that head's 512-row block.
    begin = 256 if gate else 0
    rows = tuple((row // 256) * 512 + begin + (row % 256) for row in range(6144))
    return prefix, rows


def packed_object_plan(object_name: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Verifier-local source row plan for every packed artifact object."""

    if object_name == "text/token_embedding":
        return (_all_rows("model.language_model.embed_tokens", 248320),)
    if object_name == "text/output_head":
        return (_all_rows("lm_head", 248320),)
    if object_name == "text/draft_head":
        raise ValueError("the draft head plan depends on the derived shortlist")
    parts = object_name.split("/")
    if len(parts) < 4 or parts[0] != "text" or parts[1] != "layers":
        raise ValueError(f"no packed verifier plan for {object_name}")
    layer = int(parts[2])
    suffix = "/".join(parts[3:])
    prefix = f"model.language_model.layers.{layer}."
    if suffix == "attention/query_key":
        return (
            _qproj_part(prefix + "self_attn.q_proj", False),
            _all_rows(prefix + "self_attn.k_proj", 1024),
        )
    if suffix == "attention/gate_value":
        return (
            _qproj_part(prefix + "self_attn.q_proj", True),
            _all_rows(prefix + "self_attn.v_proj", 1024),
        )
    if suffix == "attention/output":
        return (_all_rows(prefix + "self_attn.o_proj", 5120),)
    if suffix == "gdn/query_key":
        return ((prefix + "linear_attn.in_proj_qkv", tuple(range(4096))),)
    if suffix == "gdn/value_z":
        return (
            (prefix + "linear_attn.in_proj_qkv", tuple(range(4096, 10240))),
            _all_rows(prefix + "linear_attn.in_proj_z", 6144),
        )
    if suffix == "gdn/output":
        return (_all_rows(prefix + "linear_attn.out_proj", 5120),)
    if suffix == "mlp/gate_up":
        return (
            _all_rows(prefix + "mlp.gate_proj", 17408),
            _all_rows(prefix + "mlp.up_proj", 17408),
        )
    if suffix == "mlp/down":
        return (_all_rows(prefix + "mlp.down_proj", 5120),)
    raise ValueError(f"no packed verifier plan for {object_name}")


def _signed_mask(bits: int, total_bytes: int) -> torch.Tensor:
    period = _SIGN_MASK_PERIOD[bits]
    unit = torch.tensor(list(period), dtype=torch.uint8)
    repeats = (total_bytes + len(period) - 1) // len(period)
    return unit.repeat(repeats)[:total_bytes]


def _source_rows(
    reader: ShardReader,
    prefix: str,
    rows: Sequence[int],
    bits: int,
) -> torch.Tensor:
    payload = reader.get(prefix + ".weight_packed").contiguous().view(torch.uint8)
    row_bytes = payload.shape[1]
    selected = payload.index_select(0, torch.tensor(list(rows), dtype=torch.long))
    return (selected ^ _signed_mask(bits, row_bytes)).reshape(len(rows), -1)


def _source_scales(reader: ShardReader, prefix: str, rows: Sequence[int]) -> torch.Tensor:
    scales = reader.get(prefix + ".weight_scale")
    return scales.index_select(0, torch.tensor(list(rows), dtype=torch.long))


def _draft_plan(
    token_ids: torch.Tensor,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return ("lm_head", tuple(int(value) for value in token_ids.tolist())),


def verify_packed_object(
    artifact: Artifact,
    reader: ShardReader,
    object_name: str,
    token_ids: torch.Tensor,
) -> tuple[int, int, int, int, float]:
    """Compare one packed artifact object with a verifier-local source decode."""

    tensor = artifact.find(object_name)
    if not isinstance(tensor, TensorObject):
        _contract_error(f"{object_name} is not a tensor")
    numeric = get_format(tensor.format)
    if not hasattr(numeric, "bits"):
        _contract_error(f"{object_name} is not a grouped quantized object")
    bits = numeric.bits
    group_size = numeric.group_size
    if object_name == "text/draft_head":
        plan = _draft_plan(token_ids)
    else:
        plan = packed_object_plan(object_name)
    pieces = [_source_rows(reader, prefix, rows, bits) for prefix, rows in plan]
    expected_base = torch.cat(pieces, dim=0)
    scale_pieces = [_source_scales(reader, prefix, rows) for prefix, rows in plan]
    expected_scales = torch.cat(scale_pieces, dim=0)

    geometry = row_split_geometry(tensor.format, tensor.shape)
    stored_scales, stored_codes = decode_row_split_codes(
        artifact.payload(tensor), tensor.format, tensor.shape
    )
    stored_base = torch.tensor(
        bytearray(artifact.payload(tensor)[: geometry.base_bytes]), dtype=torch.uint8
    ).reshape(tensor.shape[0], -1)
    if stored_base.shape != expected_base.shape:
        _contract_error(
            f"{object_name}: stored base plane {tuple(stored_base.shape)} != "
            f"{tuple(expected_base.shape)}"
        )
    if not torch.equal(stored_base, expected_base):
        _contract_error(f"{object_name}: stored codes differ from the source")

    converted = expected_scales.float().half()
    rounded = expected_scales.float() != converted.float()
    error = (expected_scales.float() - converted.float()).abs()
    if bool((error > 2.0**-25).any()):
        _contract_error(f"{object_name}: a stored scale exceeds the rounding bound")
    if bool((rounded & (converted.float().abs() >= 2.0**-14)).any()):
        _contract_error(f"{object_name}: a non-subnormal scale is not binary16-exact")
    if not torch.equal(stored_scales.view(torch.int16), converted.view(torch.int16)):
        _contract_error(f"{object_name}: stored scales differ from the source words")
    if stored_codes.shape[0] != tensor.shape[0]:
        _contract_error(f"{object_name}: stored code rows are inconsistent")
    return (
        tensor.shape[0],
        tensor.shape[0] * (tensor.shape[1] // group_size),
        1,
        int(rounded.sum()),
        float(error.max()) if error.numel() else 0.0,
    )


def verify_packed_objects(
    artifact: Artifact,
    reader: ShardReader,
    token_ids: torch.Tensor,
) -> PackedSummary:
    objects = 0
    rows = 0
    groups = 0
    rounded = 0
    max_error = 0.0
    packed_names = _packed_names()
    for tensor in artifact.objects:
        if not isinstance(tensor, TensorObject) or tensor.name not in packed_names:
            continue
        object_name = tensor.name
        row_count, group_count, _base_ok, rounded_count, error = verify_packed_object(
            artifact, reader, object_name, token_ids
        )
        objects += 1
        rows += row_count
        groups += group_count
        rounded += rounded_count
        max_error = max(max_error, error)
    return PackedSummary(
        objects=objects,
        rows=rows,
        groups=groups,
        base_bytes_equal=objects,
        scales_equal=objects,
        scales_rounded=rounded,
        max_scale_error=max_error,
    )


def _packed_names() -> frozenset[str]:
    names = {"text/token_embedding", "text/output_head", "text/draft_head"}
    for layer in range(64):
        prefix = f"text/layers/{layer}/"
        names.update(
            {
                prefix + "attention/query_key",
                prefix + "attention/gate_value",
                prefix + "attention/output",
                prefix + "gdn/query_key",
                prefix + "gdn/value_z",
                prefix + "gdn/output",
                prefix + "mlp/gate_up",
                prefix + "mlp/down",
            }
        )
    return frozenset(names)


def _reader_for(name: str, gsq_reader: ShardReader, official_reader: ShardReader) -> ShardReader:
    return official_reader if name.startswith("mtp.") else gsq_reader


def _load_draft_token_ids(artifact: Artifact) -> torch.Tensor:
    obj = artifact.find("text/draft_head_token_ids")
    if not isinstance(obj, TensorObject):
        _contract_error("draft token-id object is not a tensor")
    token_ids = decode_direct(artifact.payload(obj), obj.format, obj.shape)
    family_verify.validate_draft_token_ids(token_ids)
    return token_ids


def _first_source_name(expression: object) -> str:
    sources = family_recipe.expression_sources(expression)
    if not sources:
        raise ValueError("recipe has no source tensors")
    return sources[0].name

def _verify_resources_and_frontend(
    artifact: Artifact,
    gsq_dir: Path,
    official_dir: Path,
) -> tuple[str, str]:
    payloads: dict[str, bytes] = {}
    for spec in inventory.RESOURCE_SPECS:
        obj = artifact.find(spec.name)
        if not isinstance(obj, ResourceObject):
            _contract_error(f"{spec.name} is not a resource")
        payload = bytes(artifact.payload(obj))
        filename = spec.name.removeprefix("frontend/")
        expected_hash = convert_gsq3.base_convert.OFFICIAL_RESOURCE_SHA256[spec.name]
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            _contract_error(f"frontend resource does not match the official pin: {spec.name}")
        for root in (official_dir, gsq_dir):
            candidate = root / filename
            if candidate.is_file():
                if payload != candidate.read_bytes():
                    _contract_error(f"frontend resource differs from source: {spec.name}")
                break
        else:
            _contract_error(f"frontend resource has no source candidate: {spec.name}")
        payloads[filename] = payload

    try:
        from transformers import AutoProcessor, GenerationConfig
    except ModuleNotFoundError:
        # The resource bytes and hashes are the artifact contract; the parser check needs the
        # optional conversion toolchain and is reported as unavailable rather than silently
        # passing.
        return "transformers-absent", "transformers-absent"

    with tempfile.TemporaryDirectory(prefix="ninfer-gsq3-frontend-") as temporary:
        directory = Path(temporary)
        for filename, payload in payloads.items():
            (directory / filename).write_bytes(payload)
        processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        generation_config = GenerationConfig.from_pretrained(
            directory, local_files_only=True
        )
        if getattr(processor, "tokenizer", None) is None:
            _contract_error("AutoProcessor did not construct its tokenizer")
        return type(processor).__name__, type(generation_config).__name__


def verify_payloads(
    artifact: Artifact,
    gsq_dir: str | Path,
    official_dir: str | Path,
    device: str | torch.device = "cpu",
) -> PayloadSummary:
    """Verify stored tensors against the pinned source shards."""

    gsq_source_dir = Path(gsq_dir)
    official_source_dir = Path(official_dir)
    draft = convert_gsq3.draft_head.compute_shortlist(
        PROJECT_ROOT / convert_gsq3.draft_head.DEFAULT_RANKING, gsq_source_dir
    )
    token_ids = _load_draft_token_ids(artifact)
    expected_ids = convert_gsq3.draft_head.materialize_draft_head_token_ids(draft)
    if not torch.equal(token_ids, expected_ids):
        _contract_error("stored draft token IDs differ from the registered shortlist")

    target = torch.device(device)
    with ShardReader(gsq_source_dir) as gsq_reader, convert_gsq3.open_official_reader(
        official_source_dir
    ) as official_reader:
        packed = verify_packed_objects(artifact, gsq_reader, token_ids)

        direct_probes = 0
        for object_name in DIRECT_PROBE_OBJECTS:
            family_verify._verify_direct_probe(artifact, gsq_reader, object_name)
            direct_probes += 1

        quantized_rows = 0
        quantized_groups = 0
        for object_name in QUANTIZED_PROBE_OBJECTS:
            obj = artifact.find(object_name)
            if not isinstance(obj, TensorObject) or len(obj.shape) != 2:
                _contract_error(f"quantized probe is not a matrix: {object_name}")
            rows = family_verify._three_indices(obj.shape[0])
            expression = recipe.RECIPES_BY_NAME[object_name].expression
            reader = _reader_for(
                _first_source_name(expression),
                gsq_reader,
                official_reader,
            )
            source_rows = family_verify._materialize_rows(
                expression,
                rows,
                family_verify._SourceSlices(reader),
                token_ids,
            )
            quantized_groups += family_verify.verify_quantized_rows(
                artifact.payload(obj),
                obj.format,
                obj.shape,
                rows,
                source_rows,
                target,
            )
            quantized_rows += len(rows)

    processor_class, generation_config_class = _verify_resources_and_frontend(
        artifact, gsq_source_dir, official_source_dir
    )
    return PayloadSummary(
        direct_probes=direct_probes,
        quantized_probes=len(QUANTIZED_PROBE_OBJECTS),
        quantized_rows=quantized_rows,
        quantized_groups=quantized_groups,
        packed=packed,
        draft_rows=int(token_ids.numel()),
        resources=len(inventory.RESOURCE_SPECS),
        processor_class=processor_class,
        generation_config_class=generation_config_class,
    )


def verify_artifact(
    artifact: Artifact,
    gsq_dir: str | Path,
    official_dir: str | Path,
    device: str | torch.device = "cpu",
) -> VerificationSummary:
    structure = validate_structure(artifact)
    payload = verify_payloads(artifact, gsq_dir, official_dir, device)
    return VerificationSummary(structure=structure, payload=payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a Qwen3.8-27B GSQ3 NInfer artifact against its sources"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--gsq-model", type=Path, required=True)
    parser.add_argument("--official-model", type=Path, required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    with Artifact.open(arguments.artifact) as artifact:
        summary = verify_artifact(
            artifact, arguments.gsq_model, arguments.official_model, arguments.device
        )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
