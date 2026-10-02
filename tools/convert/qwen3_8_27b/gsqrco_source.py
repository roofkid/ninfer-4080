"""GGUF value source for the Qwen3.8-27B GSQ-RCO IQ3_S port.

Tensors are dequantized with llama.cpp's reference implementation (the ``gguf``
package), assembled into the registered artifact row orders, and returned as
rank-2 float32 tensors in the NInfer orientation ``[rows, k]``.  Nothing here
writes an artifact; :mod:`convert_gsqrco` pairs this source with the inventory
formats and the shared grouped quantizer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader
from gguf.quants import dequantize

from . import inventory_gsqrco as inventory
from . import recipe_gsq3




def reorder_v_heads(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Invert llama.cpp's grouped->tiled V-head reorder along ``dim``.

    The GGUF stores tiled ``[group, K-head, dim]``; the artifact expects the
    source's grouped ``[K-head, group, dim]``.  Swapping the two head axes is
    self-inverse, so the same helper also describes the forward direction."""

    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    head_axes = [NUM_V_PER_K, NUM_K_HEADS, HEAD_V_DIM]
    new_shape = shape[:dim] + head_axes + shape[dim + 1 :]
    reshaped = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return reshaped.permute(*perm).contiguous().reshape(*shape)
def gguf_tensor_types(path: str | Path) -> dict[str, str]:
    """Open one GGUF and return ``{tensor name: GGML type name}``."""

    reader = GGUFReader(str(path))
    return {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}


# Qwen3.8 GDN geometry; the GGUF converter reorders V heads from grouped
# ([G0_v0..v2, G1_v0..v2, ...]) to tiled ([G0_v0, G1_v0, ..., G0_v1, ...]) order for
# ggml broadcast, so the same self-inverse permutation recovers the artifact order.
NUM_K_HEADS = 16
NUM_V_PER_K = 3
HEAD_V_DIM = 128
class GgufSource:
    """Dequantizing reader over one GGUF file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._reader = GGUFReader(str(path))
        self._tensors = {tensor.name: tensor for tensor in self._reader.tensors}

    @property
    def tensor_types(self) -> dict[str, str]:
        return {name: tensor.tensor_type.name for name, tensor in self._tensors.items()}

    def matrix(self, name: str) -> torch.Tensor:
        tensor = self._tensors[name]
        values = dequantize(np.asarray(tensor.data), tensor.tensor_type)
        if values.ndim != 2:
            raise ValueError(f"{name}: expected a rank-2 tensor, got {values.shape}")
        return torch.from_numpy(np.ascontiguousarray(values))

    def object_matrix(self, name: str) -> torch.Tensor:
        """Materialize one ported artifact object in artifact row order."""

        sources = inventory.object_source_tensors(name)
        if sources is None:
            raise ValueError(f"{name} is not a ported text-core object")
        parts = name.split("/")
        if name == "text/token_embedding" or name == "text/output_head":
            return self.matrix(sources[0])
        suffix = "/".join(parts[3:])
        if suffix == "attention/query_key":
            query_rows, _ = recipe_gsq3.attention_indices()
            query = self.matrix(sources[0])
            return torch.cat((query[list(query_rows)], self.matrix(sources[1])), dim=0)
        if suffix == "attention/gate_value":
            _, gate_rows = recipe_gsq3.attention_indices()
            query = self.matrix(sources[0])
            return torch.cat((query[list(gate_rows)], self.matrix(sources[1])), dim=0)
        if suffix == "attention/output":
            return self.matrix(sources[0])
        if suffix == "gdn/query_key":
            return self.matrix(sources[0])[:4096].clone()
        if suffix == "gdn/value_z":
            qkv = self.matrix(sources[0])
            v = reorder_v_heads(qkv[4096:10240], 0)
            z = reorder_v_heads(self.matrix(sources[1]), 0)
            return torch.cat((v, z), dim=0)
        if suffix == "gdn/output":
            return reorder_v_heads(self.matrix(sources[0]), 1)
        if suffix == "mlp/gate_up":
            return torch.cat((self.matrix(sources[0]), self.matrix(sources[1])), dim=0)
        if suffix == "mlp/down":
            return self.matrix(sources[0])
        raise ValueError(f"{name}: no ported row assembly is registered")


__all__ = ["GgufSource", "gguf_tensor_types"]
