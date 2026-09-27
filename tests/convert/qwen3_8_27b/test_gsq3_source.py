from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from tools.convert.common.safetensors import ShardReader
from tools.convert.qwen3_8_27b import gsq3_source


def _pack_stream(codes: np.ndarray, bits: int, *, signed: bool) -> bytes:
    """Independent dense LSB-first packer for the publisher/artifact planes."""

    values = codes.astype(np.int64).reshape(-1)
    if signed:
        values = values & ((1 << bits) - 1)
    else:
        values = values + (1 << (bits - 1))
    accumulator = 0
    for position, value in enumerate(values.tolist()):
        assert 0 <= value < (1 << bits)
        accumulator |= value << (position * bits)
    total_bytes = (values.size * bits + 7) // 8
    return accumulator.to_bytes(total_bytes, "little")


def _matrix(codes: np.ndarray, bits: int, group_size: int) -> gsq3_source.GsqMatrix:
    rows, columns = codes.shape
    assert columns % group_size == 0 and (columns * bits) % 32 == 0
    stream = np.frombuffer(_pack_stream(codes, bits, signed=False), dtype=np.uint8).copy()
    words = stream.view(np.int32).reshape(rows, columns * bits // 32)
    groups = columns // group_size
    scales = np.ones((rows, groups), dtype=np.float32).astype(np.float16)
    return gsq3_source.GsqMatrix(
        prefix="test",
        rows=rows,
        columns=columns,
        bits=bits,
        group_size=group_size,
        packed=torch.from_numpy(words.copy()),
        scales=torch.from_numpy(scales).to(torch.bfloat16),
    )


def _first_group(codes: np.ndarray) -> np.ndarray:
    row = np.zeros((codes.shape[0], 128), dtype=np.int64)
    row[:, : codes.shape[1]] = codes
    return row


def test_known_vector_matches_registered_artifact_plane() -> None:
    codes = np.array([[-4, -3, -2, -1, 0, 1, 2, 3]], dtype=np.int64)
    matrix = _matrix(_first_group(codes), 3, 128)
    plane = gsq3_source.artifact_plane_bytes(matrix)
    assert plane.shape == (1, 1, 48)
    assert bytes(plane.reshape(-1)[:3]) == b"\xac\x8f\x68"
    assert bytes(plane.reshape(-1)[:3]) == _pack_stream(_first_group(codes), 3, signed=True)[:3]


def test_source_decode_uses_the_shifted_grid() -> None:
    codes = np.array([[-4, -3, -2, -1, 0, 1, 2, 3]], dtype=np.int64)
    matrix = _matrix(_first_group(codes), 3, 128)
    assert torch.equal(
        gsq3_source.source_codes(matrix)[:, :8], torch.from_numpy(codes.astype(np.int8))
    )

@pytest.mark.parametrize(("bits", "group_size"), ((3, 128), (4, 64)))
def test_round_trip_preserves_codes_and_sign_flip(bits: int, group_size: int) -> None:
    rng = np.random.default_rng(7)
    codes = rng.integers(-(1 << (bits - 1)), 1 << (bits - 1), size=(5, group_size * 2))
    matrix = _matrix(codes, bits, group_size)
    assert torch.equal(gsq3_source.source_codes(matrix), torch.from_numpy(codes.astype(np.int8)))
    plane = gsq3_source.artifact_plane_bytes(matrix)
    assert plane.shape == (5, 2, group_size * bits // 8)
    expected = _pack_stream(codes, bits, signed=True)
    assert plane.reshape(5, -1).numpy().tobytes() == expected


def test_row_selection_and_concatenation_preserve_payloads() -> None:
    rng = np.random.default_rng(11)
    codes = rng.integers(-4, 4, size=(6, 256))
    matrix = _matrix(codes, 3, 128)
    selected = matrix.select_rows([4, 1])
    assert torch.equal(
        gsq3_source.source_codes(selected), torch.from_numpy(codes[[4, 1]].astype(np.int8))
    )
    joined = gsq3_source.concat_rows((matrix.select_rows([0, 1]), matrix.select_rows([2])))
    assert joined.rows == 3
    assert torch.equal(
        gsq3_source.source_codes(joined), torch.from_numpy(codes[[0, 1, 2]].astype(np.int8))
    )
    with pytest.raises(gsq3_source.GsqSourceError):
        matrix.select_rows([6])


def test_scale_conversion_is_exact_above_the_subnormal_range() -> None:
    normal = torch.tensor([1.5, -0.25, 6.103515625e-05], dtype=torch.bfloat16)
    converted, audit = gsq3_source.convert_scales(normal)
    assert torch.equal(converted.float(), normal.float())
    assert audit.rounded == 0
    assert audit.max_abs_error == 0.0


def test_scale_conversion_accounts_for_binary16_subnormal_rounding() -> None:
    # A bf16 word half a binary16 subnormal step above the grid; the pinned
    # source contains 218 such words, all with |scale| < 2**-14.
    subnormal = torch.tensor([3.784894943e-06, -2.294778824e-06], dtype=torch.bfloat16)
    converted, audit = gsq3_source.convert_scales(subnormal)
    assert audit.rounded == 2
    assert audit.max_abs_error <= 2.0**-25
    assert bool((converted.abs() < 2.0**-14).all())
    assert torch.equal(
        converted,
        torch.tensor([3.814697265625e-06, -2.2649765014648438e-06], dtype=torch.float16),
    )


def test_scale_conversion_rejects_nonfinite_and_out_of_range() -> None:
    with pytest.raises(gsq3_source.GsqSourceError):
        gsq3_source.convert_scales(torch.tensor([float("inf")], dtype=torch.bfloat16))
    with pytest.raises(gsq3_source.GsqSourceError):
        gsq3_source.convert_scales(torch.tensor([1.0e6], dtype=torch.bfloat16))


def test_encode_payload_matches_registered_decoder(tmp_path: Path) -> None:
    from tools.artifact.layouts import decode_row_split_codes

    rng = np.random.default_rng(3)
    codes = rng.integers(-4, 4, size=(3, 5120))
    matrix = _matrix(codes, 3, 128)
    payload, audit = gsq3_source.encode_payload(matrix)
    assert audit.words == 3 * 40
    decoded_scales, decoded_codes = decode_row_split_codes(payload, "Q3G128_F16S", (3, 5120))
    assert torch.equal(decoded_scales.float(), matrix.scales.float())
    assert torch.equal(
        decoded_codes.reshape(3, 5120), torch.from_numpy(codes.astype(np.int8))
    )


def test_read_matrix_accepts_a_synthetic_source_and_rejects_geometry(tmp_path: Path) -> None:
    codes = np.zeros((2, 256), dtype=np.int64)
    codes[0, 0] = -4
    codes[1, 255] = 3
    stream = np.frombuffer(_pack_stream(codes, 3, signed=False), dtype=np.uint8)
    words = torch.from_numpy(stream.view(np.int32).reshape(2, 24).copy())
    scales = torch.zeros((2, 2), dtype=torch.bfloat16)
    scales[0, 0] = 0.5
    path = tmp_path / "model.safetensors"
    save_file(
        {
            "layer.weight_packed": words,
            "layer.weight_scale": scales,
            "layer.weight_shape": torch.tensor([2, 256], dtype=torch.int64),
        },
        str(path),
    )
    with ShardReader.from_file(path) as reader:
        matrix = gsq3_source.read_matrix(reader, "layer", bits=3, group_size=128)
        assert matrix.rows == 2 and matrix.columns == 256
        assert torch.equal(
            gsq3_source.source_codes(matrix), torch.from_numpy(codes.astype(np.int8))
        )
        payload, _ = gsq3_source.encode_payload(matrix)
        assert len(payload) > 0

    save_file(
        {
            "layer.weight_packed": words[:, :20].contiguous(),
            "layer.weight_scale": scales,
            "layer.weight_shape": torch.tensor([2, 256], dtype=torch.int64),
        },
        str(path),
    )
    with ShardReader.from_file(path) as reader:
        with pytest.raises(gsq3_source.GsqSourceError, match="weight_packed"):
            gsq3_source.read_matrix(reader, "layer", bits=3, group_size=128)
