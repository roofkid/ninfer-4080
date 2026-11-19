#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// YaRN frequency/attention scaling parameters. `attention_factor` <= 0 infers the reference
// default 0.1 * log(factor) + 1. The same scaled inverse frequencies are used on all MRoPE axes.
struct YarnRopeParameters {
    float factor                            = 2.0F;
    float original_max_position_embeddings  = 262144.0F;
    float beta_fast                         = 32.0F;
    float beta_slow                         = 1.0F;
    float attention_factor                  = 0.0F;
    bool truncate                           = true;
};

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. Text geometry
 * 16/4 uses the generic domain. The YaRN overload applies the ramp-interpolated inverse
 * frequencies and the attention factor at every position; it evaluates the frequency ramp and
 * phase in FP64 so 256K/512K positions retain the naive-oracle accuracy. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          cudaStream_t stream);

// YaRN form. In addition to the base inverse frequencies, the ramp-interpolated YaRN frequency
// table is evaluated in FP32 and cos/sin are multiplied by the attention factor. The layout and
// mutation contract are identical to the unscaled form.
void rope(const Tensor& positions, int rotary_dim, float theta, const YarnRopeParameters& yarn,
          Tensor& q, Tensor& k, cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x, cudaStream_t stream);

void rope(const Tensor& positions, int rotary_dim, float theta, const YarnRopeParameters& yarn,
          Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
