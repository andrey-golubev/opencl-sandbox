#pragma once

#include <cstdint>
#include <stdexcept>

#include "matrix.h"

// interface:
RGBMat eltwise_sum_cpp(const RGBMat& a, const RGBMat& b) {
    RGBMat result(a);

    size_t rows = a.data.size(), cols = a.data[0].size();

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            auto& res_pixel = result.data[i][j];
            const auto& a_pixel = a.data[i][j];
            const auto& b_pixel = b.data[i][j];

            for (size_t chan = 0; chan < res_pixel.size(); ++chan) {
                res_pixel[chan] = a_pixel[chan] + b_pixel[chan];
            }
        }
    }
    return result;
}
