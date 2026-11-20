from __future__ import annotations

import torch

from tools.convert.sensenova_loop_9b.awq import dequantize_gemm, dequantize_row_major

_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _pack_rows(values: torch.Tensor, order: tuple[int, ...] | None) -> torch.Tensor:
    rows, cols = values.shape
    packed = torch.zeros((rows, cols // 8), dtype=torch.int32)
    for lane in range(8):
        source = lane if order is None else order[lane]
        packed |= values[:, source::8].to(torch.int32) << (4 * lane)
    return packed


def test_gemm_decode_matches_fp64_oracle() -> None:
    torch.manual_seed(7)
    out_features, in_features, group_size = 40, 256, 128
    groups = in_features // group_size
    codes = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    zeros = torch.randint(0, 16, (groups, out_features), dtype=torch.int32)
    scales = torch.rand(groups, out_features, dtype=torch.float16) + 0.001

    qweight = _pack_rows(codes, _PACK_ORDER)
    qzeros = _pack_rows(zeros, _PACK_ORDER)
    actual = dequantize_gemm(qweight, qzeros, scales, group_size)

    expect = (
        codes.reshape(groups, group_size, out_features).to(torch.float64)
        - zeros.to(torch.float64).unsqueeze(1)
    ) * scales.to(torch.float64).unsqueeze(1)
    expect = expect.reshape(in_features, out_features).T
    assert torch.equal(actual, expect.to(torch.float32))


def test_row_major_embedding_matches_fp64_oracle() -> None:
    torch.manual_seed(11)
    vocab, hidden, group_size = 24, 1024, 128
    groups = hidden // group_size
    codes = torch.randint(0, 16, (vocab, hidden), dtype=torch.int32)
    zeros = torch.randint(0, 16, (vocab, groups), dtype=torch.int32)
    scales = torch.rand(vocab, groups, dtype=torch.float16) + 0.001

    qweight = _pack_rows(codes, None)
    qzeros = _pack_rows(zeros, None)
    actual = dequantize_row_major(qweight, qzeros, scales, group_size)

    expect = (
        codes.reshape(vocab, groups, group_size).to(torch.float64)
        - zeros.to(torch.float64).unsqueeze(-1)
    ) * scales.to(torch.float64).unsqueeze(-1)
    assert torch.equal(actual, expect.reshape(vocab, hidden).to(torch.float32))
