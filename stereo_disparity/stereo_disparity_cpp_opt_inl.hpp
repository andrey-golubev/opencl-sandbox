#pragma once

#include <cmath>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>

#include "common/utils.hpp"

#include "stereo_common.hpp"

#ifndef CV_SIMD
#error "OpenCV's CV_SIMD undefined, impossible to use this header."
#endif

namespace stereo_cpp_opt {
int fix(int v, int max_v) {
    if (v < 0) {
        return -v;
    } else if (v > max_v) {
        const int diff = (v - max_v);
        return max_v - diff;
    }
    return v;
}

constexpr const int UINT8_NLANES = cv::v_uint8::nlanes;  // SIMD specific constant

// List of functions:

// copies matrix with border for window_size
cv::Mat copy_make_border(const cv::Mat& in, int window_size);

// copy row
void _kernel_copy(const uchar* in, uchar* out, int cols) {
    int l = 0;
    for (; l <= cols - UINT8_NLANES; l += UINT8_NLANES) {
        cv::v_uint8x16 p = cv::v_load(&in[l]);
        cv::v_store(&out[l], p);
    }

    // tail:
    if (l < cols) {
        l = cols - UINT8_NLANES;
        cv::v_uint8x16 p = cv::v_load(&in[l]);
        cv::v_store(&out[l], p);
    }
}

void _kernel_copy_make_border(const uchar* in_row, uchar* out_row, int cols, int border) {
    // fast copy within in_row region
    _kernel_copy(in_row, out_row + border, cols);

    // slow copy border pixels
    for (int j = 1; j <= border; ++j) {
        // do reflect101 copy
        out_row[border - j] = in_row[j];
        out_row[cols - 1 + border + j] = in_row[cols - 1 - j];
    }
}

cv::Mat copy_make_border(const cv::Mat& in, int window_size) {
    REQUIRE(in.type() == CV_8UC1);
    const int rows = in.rows, cols = in.cols;
    REQUIRE(cols >= UINT8_NLANES);
    const int border = (window_size - 1) / 2;

    cv::Mat out = cv::Mat::zeros(cv::Size(cols + border * 2, rows + border * 2), in.type());

    // copy input rows
    for (int i = 0; i < rows; ++i) {
        const uchar* in_row = (in.data + i * cols);
        uchar* out_row = (out.data + (i + border) * (cols + border * 2));
        _kernel_copy_make_border(in_row, out_row, cols, border);
    }

    // copy reflected rows (reflect101)
    for (int j = 1; j <= border; ++j) {
        // top
        {
            const uchar* in_row = (in.data + j * cols);
            uchar* out_row = (out.data + (border - j) * (cols + border * 2));
            _kernel_copy_make_border(in_row, out_row, cols, border);
        }
        // bottom
        {
            const uchar* in_row = (in.data + (rows - 1 - j) * cols);
            uchar* out_row = (out.data + (rows - 1 + border + j) * (cols + border * 2));
            _kernel_copy_make_border(in_row, out_row, cols, border);
        }
    }

    return out;
}
}  // namespace stereo_cpp_opt
