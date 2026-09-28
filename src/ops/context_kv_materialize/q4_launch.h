#pragma once

#include "ninfer/ops/context_kv_materialize.h"

#include <array>

namespace ninfer::ops::detail {

// Composed Q4 companion route for context K/V materialization. A qualified Q4 linear projects each
// layer's key/value rows into one caller-owned BF16 scratch [1024,W*B]; the two store kernels then
// apply the key norm and RoPE, and the value BF16->FP16 boundary, at the same cyclic-ring
// destinations as the fused W8 route. Key raw values are represented in BF16, which the public Op
// contract admits as private projection staging precision.
void context_kv_materialize_q4_launch(
    const Tensor& context, const Tensor& positions, const Tensor& counts, const Tensor& state_slots,
    const std::array<ContextKVMaterializeLayerView, kContextKVMaterializeLayers>& layers,
    ContextKVMaterializeExecutionEnvelope envelope, const Tensor& scratch, cudaStream_t stream);

} // namespace ninfer::ops::detail
