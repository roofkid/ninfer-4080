"""GSQ ``pack-quantized`` source adapter for the Qwen3.8-27B 3-bit artifact.

The pinned publisher release (D2) stores every quantized matrix as three
safetensors objects:

* ``<prefix>.weight_packed`` -- ``I32 [N, K*bits/32]``: a dense little-endian
  bitstream of unsigned code words, 32 bits per stored word, code ``j`` at
  global bits ``[bits*j, bits*j + bits)``;
* ``<prefix>.weight_scale`` -- ``BF16 [N, K/group]``: one signed multiplier per
  code group;
* ``<prefix>.weight_shape`` -- ``I64 [2]``: the logical ``[N, K]`` shape.

``compressed-tensors`` ``pack_to_int32`` shifts the signed grid
``[-2^(bits-1), 2^(bits-1)-1]`` into the unsigned domain, so the represented
matrix is ``w = (u - 2^(bits-1)) * scale``.  The registered NInfer
``Q3G128_F16S``/``Q4G64_F16S`` row-split planes store the signed code in two's
complement.  The two encodings differ exactly by the sign bit of every code,
i.e. a fixed XOR mask over the dense stream.  This module owns that
transformation and the scale-width conversion; it never requantizes a code.

The bf16 multiplier plane is stored as binary16 in the artifact.  A bf16 value
is binary16-exact whenever its magnitude reaches the binary16 normal range;
below ``2**-14`` the conversion may round at the binary16 subnormal step.  The
adapter asserts a hard ``2**-25`` absolute bound and that every rounded word is
subnormal, and reports the audit so the conversion report can state it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from tools.artifact.layouts import RowPlanes, assemble_row_planes, row_split_geometry
from tools.artifact.numeric import Q3G128_F16S, Q4G64_F16S, get_format
from tools.convert.common.safetensors import ShardReader


GSQ_REPOSITORY = "ISTA-DASLab/Qwen3.8-27B-3Bit-GSQ"
GSQ_REVISION = "b5ce0b76f60020a875dee4f6ec9d934cca4121e4"
OFFICIAL_REPOSITORY = "Qwen/Qwen3.8-27B"
OFFICIAL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"

# The publisher's own packing is LSB-first from the start of the row stream.
# These masks flip bit (bits - 1) of every code, which is the exact difference
# between the shifted source grid and the registered two's-complement plane.
_SIGN_BIT_PERIODS = {
    3: bytes((0x24, 0x49, 0x92)),
    4: bytes((0x88,)),
}

_FORMAT_BY_GEOMETRY = {
    (3, 128): Q3G128_F16S,
    (4, 64): Q4G64_F16S,
}

_FP16_MIN_NORMAL = 2.0**-14
_FP16_MAX = 65504.0
_SCALE_ROUNDING_BOUND = 2.0**-25


class GsqSourceError(ValueError):
    """The source shard does not satisfy the pinned GSQ contract."""


@dataclass(frozen=True, slots=True)
class GsqMetadata:
    prefix: str
    rows: int
    columns: int
    bits: int
    group_size: int


@dataclass(frozen=True, slots=True)
class ScaleAudit:
    """Binary16 conversion accounting for one matrix."""

    words: int
    rounded: int
    max_abs_error: float

    def merge(self, other: "ScaleAudit") -> "ScaleAudit":
        return ScaleAudit(
            words=self.words + other.words,
            rounded=self.rounded + other.rounded,
            max_abs_error=max(self.max_abs_error, other.max_abs_error),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "words": self.words,
            "rounded": self.rounded,
            "max_abs_error": self.max_abs_error,
        }


@dataclass(frozen=True, slots=True)
class GsqMatrix:
    """One logical ``[N, K]`` matrix in the source packing plus its scales."""

    prefix: str
    rows: int
    columns: int
    bits: int
    group_size: int
    packed: torch.Tensor  # int32 [rows, columns*bits//32]
    scales: torch.Tensor  # bfloat16 [rows, columns//group_size]

    @property
    def groups_per_row(self) -> int:
        return self.columns // self.group_size

    @property
    def words_per_row(self) -> int:
        return self.packed.shape[1]

    def select_rows(self, rows: Sequence[int] | torch.Tensor) -> "GsqMatrix":
        if isinstance(rows, torch.Tensor):
            if rows.dim() != 1 or rows.numel() == 0:
                raise GsqSourceError("row selection requires a nonempty one-dimensional tensor")
            index = rows.to(torch.long)
        else:
            if not rows:
                raise GsqSourceError("row selection requires at least one row")
            index = torch.tensor(list(rows), dtype=torch.long)
        if int(index.min()) < 0 or int(index.max()) >= self.rows:
            raise GsqSourceError(f"{self.prefix}: row selection is outside the matrix")
        return GsqMatrix(
            prefix=self.prefix,
            rows=int(index.numel()),
            columns=self.columns,
            bits=self.bits,
            group_size=self.group_size,
            packed=self.packed.index_select(0, index).contiguous(),
            scales=self.scales.index_select(0, index).contiguous(),
        )


def _require_dtype(tensor: torch.Tensor, dtype: torch.dtype, name: str) -> None:
    if tensor.dtype != dtype:
        raise GsqSourceError(f"{name}: expected {dtype}, got {tensor.dtype}")


def read_metadata(reader: ShardReader, prefix: str, *, bits: int, group_size: int) -> GsqMetadata:
    """Read and validate the three source objects without loading payloads."""

    if bits not in _SIGN_BIT_PERIODS:
        raise GsqSourceError(f"{prefix}: unsupported source code width {bits}")
    shape_name = prefix + ".weight_shape"
    metadata = reader.metadata((shape_name, prefix + ".weight_packed", prefix + ".weight_scale"))
    shape = metadata[shape_name]
    if shape.dtype != "I64" or shape.shape != (2,):
        raise GsqSourceError(f"{shape_name}: expected I64[2], got {shape.dtype}{shape.shape}")
    packed = metadata[prefix + ".weight_packed"]
    scales = metadata[prefix + ".weight_scale"]
    if packed.dtype != "I32" or len(packed.shape) != 2:
        raise GsqSourceError(f"{prefix}.weight_packed: expected an I32 matrix")
    if scales.dtype != "BF16" or len(scales.shape) != 2:
        raise GsqSourceError(f"{prefix}.weight_scale: expected a BF16 matrix")
    shape_value = reader.get(shape_name)
    rows, columns = (int(value) for value in shape_value.tolist())
    if rows <= 0 or columns <= 0:
        raise GsqSourceError(f"{shape_name}: nonpositive logical shape")
    if columns % group_size != 0:
        raise GsqSourceError(f"{prefix}: K={columns} is not a multiple of {group_size}")
    if (columns * bits) % 32 != 0:
        raise GsqSourceError(f"{prefix}: K={columns} does not fill int32 words")
    expected_packed = (rows, columns * bits // 32)
    if packed.shape != expected_packed:
        raise GsqSourceError(
            f"{prefix}.weight_packed: shape {packed.shape} does not match logical "
            f"{rows}x{columns} at {bits} bits"
        )
    expected_scales = (rows, columns // group_size)
    if scales.shape != expected_scales:
        raise GsqSourceError(
            f"{prefix}.weight_scale: shape {scales.shape} does not match logical "
            f"{rows}x{columns} at group {group_size}"
        )
    return GsqMetadata(prefix, rows, columns, bits, group_size)


def read_matrix(reader: ShardReader, prefix: str, *, bits: int, group_size: int) -> GsqMatrix:
    """Load one matrix payload and validate its geometry and scales."""

    meta = read_metadata(reader, prefix, bits=bits, group_size=group_size)
    packed = reader.get(prefix + ".weight_packed")
    scales = reader.get(prefix + ".weight_scale")
    _require_dtype(packed, torch.int32, prefix + ".weight_packed")
    _require_dtype(scales, torch.bfloat16, prefix + ".weight_scale")
    if not bool(torch.isfinite(scales.float()).all()):
        raise GsqSourceError(f"{prefix}.weight_scale: contains NaN or infinity")
    return GsqMatrix(
        prefix=prefix,
        rows=meta.rows,
        columns=meta.columns,
        bits=meta.bits,
        group_size=meta.group_size,
        packed=packed.contiguous(),
        scales=scales.contiguous(),
    )


def concat_rows(parts: Iterable[GsqMatrix]) -> GsqMatrix:
    """Concatenate matrices along logical rows without touching the codes."""

    items = tuple(parts)
    if not items:
        raise GsqSourceError("row concatenation requires at least one matrix")
    first = items[0]
    if any(
        item.columns != first.columns
        or item.bits != first.bits
        or item.group_size != first.group_size
        for item in items
    ):
        raise GsqSourceError("row concatenation requires one geometry")
    if len(items) == 1:
        return first
    return GsqMatrix(
        prefix="+".join(item.prefix for item in items),
        rows=sum(item.rows for item in items),
        columns=first.columns,
        bits=first.bits,
        group_size=first.group_size,
        packed=torch.cat([item.packed for item in items], dim=0).contiguous(),
        scales=torch.cat([item.scales for item in items], dim=0).contiguous(),
    )


def _sign_mask(bits: int, total_bytes: int, device: torch.device) -> torch.Tensor:
    period = _SIGN_BIT_PERIODS.get(bits)
    if period is None:
        raise GsqSourceError(f"unsupported source code width {bits}")
    unit = torch.tensor(list(period), dtype=torch.uint8, device=device)
    repeats = (total_bytes + unit.numel() - 1) // unit.numel()
    return unit.repeat(repeats)[:total_bytes]


def artifact_plane_bytes(matrix: GsqMatrix) -> torch.Tensor:
    """Return the registered row-split base plane bytes ``[rows, row_bytes]``."""

    words = matrix.packed
    if words.dim() != 2 or words.shape != (matrix.rows, matrix.words_per_row):
        raise GsqSourceError(f"{matrix.prefix}: packed plane geometry changed")
    stream = words.contiguous().view(torch.uint8)
    mask = _sign_mask(matrix.bits, stream.shape[1], stream.device)
    shifted = stream ^ mask
    groups = matrix.groups_per_row
    bytes_per_group = matrix.group_size * matrix.bits // 8
    return shifted.reshape(matrix.rows, groups, bytes_per_group)


def convert_scales(scales: torch.Tensor) -> tuple[torch.Tensor, ScaleAudit]:
    """Convert bf16 multipliers to the registered binary16 plane.

    The conversion is exact for every normal-range word.  Rounded words must be
    binary16 subnormal and are bounded by ``2**-25``; anything else is a source
    contract violation and raises instead of silently degrading the artifact.
    """

    if scales.dtype != torch.bfloat16:
        raise GsqSourceError("scale conversion requires bfloat16 source words")
    source = scales.float()
    converted = source.half()
    back = converted.float()
    if not bool(torch.isfinite(source).all()):
        raise GsqSourceError("scale plane contains NaN or infinity")
    if bool((source.abs() > _FP16_MAX).any()):
        raise GsqSourceError("scale plane exceeds the binary16 range")
    error = (source - back).abs()
    rounded = error > 0.0
    if bool((error > _SCALE_ROUNDING_BOUND).any()):
        worst = float(error.max())
        raise GsqSourceError(
            f"binary16 scale conversion exceeds the {_SCALE_ROUNDING_BOUND} bound: {worst}"
        )
    if bool((rounded & (back.abs() >= _FP16_MIN_NORMAL)).any()):
        raise GsqSourceError("a normal-range binary16 scale is not exact")
    audit = ScaleAudit(
        words=int(source.numel()),
        rounded=int(rounded.sum()),
        max_abs_error=float(error.max()) if error.numel() else 0.0,
    )
    return converted, audit


def artifact_scales(matrix: GsqMatrix) -> tuple[torch.Tensor, ScaleAudit]:
    """Return the registered binary16 scale plane ``[rows, groups]`` and audit."""

    converted, audit = convert_scales(matrix.scales)
    return converted.reshape(matrix.rows, matrix.groups_per_row), audit


def encode_payload(matrix: GsqMatrix) -> tuple[bytes, ScaleAudit]:
    """Encode one row-split payload, returning the bytes and scale audit."""

    format_spec = _FORMAT_BY_GEOMETRY.get((matrix.bits, matrix.group_size))
    if format_spec is None:
        raise GsqSourceError(
            f"{matrix.prefix}: no registered format for {matrix.bits} bits/group {matrix.group_size}"
        )
    geometry = row_split_geometry(format_spec, (matrix.rows, matrix.columns))
    if geometry.k_pad != geometry.k:
        raise GsqSourceError(f"{matrix.prefix}: K padding is not representable verbatim")
    scales, audit = artifact_scales(matrix)
    base = artifact_plane_bytes(matrix).reshape(-1).numpy().tobytes()
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    planes = RowPlanes(base, b"", scale_bytes, matrix.rows)
    payload = assemble_row_planes(planes, format_spec, matrix.columns)
    assert isinstance(payload, bytes)
    return payload, audit


def unpack_unsigned(stream: torch.Tensor, bits: int, columns: int) -> torch.Tensor:
    """Decode a dense LSB-first bitstream into unsigned code words (oracle)."""

    if bits not in _SIGN_BIT_PERIODS:
        raise GsqSourceError(f"unsupported bitstream width {bits}")
    if stream.dtype != torch.uint8 or stream.dim() != 2:
        raise GsqSourceError("bitstream decode requires a uint8 matrix")
    rows = stream.shape[0]
    shifts = torch.arange(bits, dtype=torch.int64)
    values = torch.empty((rows, columns), dtype=torch.int64)
    for code in range(columns):
        start = code * bits
        byte_index = (start + shifts) // 8
        bit_shift = (start + shifts) % 8
        if int(byte_index.max()) >= stream.shape[1]:
            raise GsqSourceError("bitstream is shorter than the requested code count")
        chunk = (stream[:, byte_index].to(torch.int64) >> bit_shift) & 1
        values[:, code] = (chunk << shifts).sum(dim=1)
    return values


def source_codes(matrix: GsqMatrix) -> torch.Tensor:
    """Decode source code words with the publisher's shifted-grid rule."""

    stream = matrix.packed.contiguous().view(torch.uint8)
    unsigned = unpack_unsigned(stream, matrix.bits, matrix.columns)
    signed = unsigned - (1 << (matrix.bits - 1))
    return signed.to(torch.int8)


def format_name(bits: int, group_size: int) -> str:
    spec = _FORMAT_BY_GEOMETRY.get((bits, group_size))
    if spec is None:
        raise GsqSourceError(f"no registered format for {bits} bits/group {group_size}")
    return get_format(spec).name


__all__ = [
    "GSQ_REPOSITORY",
    "GSQ_REVISION",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_REVISION",
    "GsqMatrix",
    "GsqMetadata",
    "GsqSourceError",
    "ScaleAudit",
    "artifact_plane_bytes",
    "artifact_scales",
    "concat_rows",
    "convert_scales",
    "encode_payload",
    "format_name",
    "read_matrix",
    "read_metadata",
    "source_codes",
    "unpack_unsigned",
]
