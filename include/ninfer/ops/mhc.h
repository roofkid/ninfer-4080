#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// mHC (multi-stream hyper-connection) routing for the SenseNova-Loop target. The reference model
// stores two residual streams and one router before the mixer and one before the MLP:
//
//   flat       = residual.flatten(streams,hidden).float()
//   normalized = flat * rsqrt(mean(flat^2, row) + 1e-6)
//   logits     = normalized @ weight.T + bias
//   pre        = sigmoid(logits[:streams])
//   post       = 2 * sigmoid(logits[streams:2*streams])
//   mix        = sinkhorn(logits[2*streams:], iterations)
//   branch     = bf16(pre) @ residual
//
// Tensor layouts (all contiguous):
//   residual BF16 [streams * hidden, tokens]
//   weight   FP32 [2 * streams + streams * streams, streams * hidden]
//   bias     FP32 [2 * streams + streams * streams]
//   branch   BF16 [hidden, tokens]
//   post     FP32 [streams, tokens]
//   mix      FP32 [streams * streams, tokens] (row-major per token)
//
// The normalization, logits, sigmoid and Sinkhorn are evaluated in FP32. `pre` and `mix` are
// rounded to BF16 before the branch/mix matmuls, and `post` is rounded to BF16 before the
// elementwise merge, matching the reference cast boundaries. Only streams=2 is registered.
void mhc_router(const Tensor& residual, const Tensor& weight, const Tensor& bias, int iterations,
                float pre_scale, Tensor& branch, Tensor& post, Tensor& mix,
                cudaStream_t stream);

// merge(residual, branch, post, mix):
//   out[s,h,t] = bf16(sum_j bf16(mix[s,j,t]) * residual[j,h,t])
//              + bf16(post[s,t]) * branch[h,t]
// The residual mix is rounded to BF16 before the additive update, as in the reference matmul.
void mhc_merge(const Tensor& residual, const Tensor& branch, const Tensor& post, const Tensor& mix,
               Tensor& out, cudaStream_t stream);

// enter(hidden, physical == start): duplicate the single BF16 residual stream into `streams`
// [streams*hidden, tokens]. leave(residual, physical == end-1): average streams in FP32 and
// publish the BF16 [hidden, tokens] single stream.
void mhc_enter(const Tensor& hidden, Tensor& streams, cudaStream_t stream);
void mhc_leave(const Tensor& streams, Tensor& hidden, cudaStream_t stream);

} // namespace ninfer::ops
