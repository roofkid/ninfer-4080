#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int q3_a16_conformance() {
    // The composed Q3 route changes projection schedule at T=8/9; every start is checked at
    // b-1/b/b+1 through the public Op.
    constexpr std::array<std::int32_t, 2> kRouteStarts{2, 9};
    constexpr std::array<std::int32_t, 12> kRouteInteriors{1, 4, 5, 8, 9, 12, 16, 64, 65, 128, 129, 256};

    int failures = 0;
    failures += ninfer::test::linear_add::run_shape(
        "Q3_A16 LinearAdd", WeightFormat::Q3G128F16S,
        ShapeCase{5120, 6144, 501U, kRouteStarts, kRouteInteriors});
    failures += ninfer::test::linear_add::run_shape(
        "Q3_A16 LinearAdd", WeightFormat::Q3G128F16S,
        ShapeCase{5120, 17408, 503U, kRouteStarts, kRouteInteriors});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q3_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q3_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q3_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
