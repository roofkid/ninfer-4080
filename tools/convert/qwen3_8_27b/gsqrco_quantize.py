"""Clipping-search grouped quantizer for the GSQ-RCO port.

The port re-quantizes already-codebook-quantized GGUF tensors a second time, so
the second grid's own error matters.  The registered max-scaled grid
(:mod:`tools.convert.common.quantize`) loses about a third more squared error
than necessary at 3 bits; this module searches a small clipping-factor ladder
per group and keeps the factor with the lowest squared error.  Binary16 scale
rounding, code selection, clamping and K padding follow the registered
transform exactly, so a decoded artifact has the same format and semantics --
only the stored scales differ.

Rows are processed in blocks so the largest registered matrices (the 248,320-row
vocabulary endpoints) stay inside a 16 GiB device.
"""

from __future__ import annotations

import numpy as np
import torch

from tools.artifact.layouts import encode_row_split, row_split_geometry
from tools.artifact.numeric import QuantFormat, get_format
from tools.convert.common.quantize import QuantizedMatrix, pick_device

DEFAULT_FACTORS = (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0)
_ROW_BLOCK = 16384
_FP16_MIN_SUBNORMAL = 2.0**-24


def _canonical_scale_words(max_abs: np.ndarray, qmax: int, factor: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw = (max_abs.astype(np.float64) * factor / float(qmax)).astype(np.float32)
        scale = raw.astype(np.float16)
    underflow = (scale == 0) & (max_abs > 0)
    if underflow.any():
        scale = scale.copy()
        scale[underflow] = np.array(_FP16_MIN_SUBNORMAL, dtype=np.float16)
    return scale


def quantize_matrix(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
    factors: tuple[float, ...] = DEFAULT_FACTORS,
) -> QuantizedMatrix:
    """Quantize logical ``[N,K]`` values with the best per-group clipping factor."""

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    if weight.dim() != 2:
        raise ValueError(f"grouped quantization requires rank 2, got {tuple(weight.shape)}")
    if not weight.dtype.is_floating_point:
        raise TypeError(f"weight must be floating point, got {weight.dtype}")
    if not weight.is_contiguous():
        weight = weight.contiguous()

    geometry = row_split_geometry(spec, weight.shape)
    target = pick_device() if device is None else pick_device(device)

    code_blocks: list[torch.Tensor] = []
    scale_blocks: list[torch.Tensor] = []
    for row_begin in range(0, geometry.n, _ROW_BLOCK):
        row_end = min(row_begin + _ROW_BLOCK, geometry.n)
        logical = weight[row_begin:row_end].detach().to(target, dtype=torch.float32)
        if geometry.k_pad != geometry.k:
            physical = torch.zeros(
                (row_end - row_begin, geometry.k_pad), dtype=torch.float32, device=target
            )
            physical[:, : geometry.k].copy_(logical)
            logical = physical
        grouped = logical.reshape(-1, geometry.groups_per_row, spec.group_size)
        max_abs = grouped.abs().amax(dim=2)
        host_max = max_abs.detach().cpu().numpy().astype(np.float32, copy=False)

        best_mse: torch.Tensor | None = None
        best_codes: torch.Tensor | None = None
        best_scales: torch.Tensor | None = None
        for factor in factors:
            scale = _canonical_scale_words(host_max, spec.qmax, factor)
            scale32 = torch.from_numpy(scale.astype(np.float32)).to(target)
            positive = scale32 > 0
            safe = torch.where(positive, scale32, torch.ones_like(scale32))
            codes = torch.clamp(
                torch.round(grouped / safe.unsqueeze(-1)), spec.qmin, spec.qmax
            ).to(torch.int8)
            codes = torch.where(positive.unsqueeze(-1), codes, torch.zeros_like(codes))
            error = (codes.float() * scale32.unsqueeze(-1) - grouped) ** 2
            mse = error.sum(dim=2)
            if best_mse is None:
                best_mse, best_codes, best_scales = mse, codes, scale32
            else:
                take = mse < best_mse
                best_mse = torch.where(take, mse, best_mse)
                best_codes = torch.where(take.unsqueeze(-1), codes, best_codes)
                best_scales = torch.where(take, scale32, best_scales)
        assert best_codes is not None and best_scales is not None
        code_blocks.append(best_codes.to("cpu"))
        scale_blocks.append(best_scales.to("cpu"))

    return QuantizedMatrix(
        codes=torch.cat(code_blocks, dim=0),
        scales=torch.cat(scale_blocks, dim=0).to(torch.float16),
    )


def quantize_and_encode(
    weight: torch.Tensor,
    format: str | QuantFormat,
    *,
    device: str | torch.device | None = None,
) -> bytes:
    """Quantize a logical matrix and encode ``row-split-k128-v1`` bytes."""

    spec = get_format(format) if isinstance(format, str) else format
    if not isinstance(spec, QuantFormat):
        raise ValueError("grouped quantization requires a quantized numeric format")
    quantized = quantize_matrix(weight, spec, device=device)
    return encode_row_split(quantized.codes, quantized.scales, spec, weight.shape)


__all__ = ["DEFAULT_FACTORS", "quantize_and_encode", "quantize_matrix"]
