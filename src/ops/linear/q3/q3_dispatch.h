#pragma once

#include "ninfer/ops/linear.h"
#include "ops/linear/q3/q3_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {

Q3Launch select_q3_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Q3Launch select_q3_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void q3_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream);

} // namespace ninfer::ops::detail
