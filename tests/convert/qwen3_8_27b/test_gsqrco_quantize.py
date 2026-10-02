from __future__ import annotations

import torch

from tools.artifact.layouts import dequantize_row_split
from tools.convert.common import quantize as registered
from tools.convert.qwen3_8_27b import gsqrco_quantize as port


def _gaussian(rows: int, columns: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, columns, generator=generator)


def _relative_error(payload: bytes, format: str, source: torch.Tensor) -> float:
    decoded = dequantize_row_split(
        payload, format, tuple(source.shape), device="cpu", dtype=torch.float32
    )
    reference = source.float()
    return (
        torch.linalg.vector_norm((decoded - reference).reshape(-1))
        / torch.linalg.vector_norm(reference.reshape(-1))
    ).item()


def test_clipping_search_beats_the_registered_grid_at_three_bits() -> None:
    source = _gaussian(512, 1024)
    registered_error = _relative_error(
        registered.quantize_and_encode(source, "Q3G128_F16S", device="cpu"),
        "Q3G128_F16S",
        source,
    )
    port_error = _relative_error(
        port.quantize_and_encode(source, "Q3G128_F16S", device="cpu"),
        "Q3G128_F16S",
        source,
    )
    assert port_error < registered_error * 0.85


def test_port_round_trip_stays_on_the_registered_geometry() -> None:
    source = _gaussian(256, 512, seed=11)
    payload = port.quantize_and_encode(source, "Q4G64_F16S", device="cpu")
    quantized = port.quantize_matrix(source, "Q4G64_F16S", device="cpu")
    assert quantized.codes.shape == (256, 512 // 64, 64)
    assert quantized.scales.shape == (256, 512 // 64)
    assert quantized.codes.dtype == torch.int8
    assert quantized.scales.dtype == torch.float16
    decoded = dequantize_row_split(
        payload, "Q4G64_F16S", (256, 512), device="cpu", dtype=torch.float32
    )
    expected = (quantized.codes.float() * quantized.scales.float().unsqueeze(-1)).reshape(
        256, 512
    )
    assert torch.equal(decoded, expected)


def test_zero_groups_decode_to_zero() -> None:
    source = torch.zeros(64, 256)
    payload = port.quantize_and_encode(source, "Q3G128_F16S", device="cpu")
    decoded = dequantize_row_split(
        payload, "Q3G128_F16S", (64, 256), device="cpu", dtype=torch.float32
    )
    assert torch.equal(decoded, source)
