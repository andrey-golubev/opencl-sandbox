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

// SIMD specific constants:
constexpr const int UINT8_NLANES = cv::v_uint8::nlanes;

// List of functions:

// copies matrix with border for window_size
cv::Mat copy_make_border(const cv::Mat& in, int window_size);
// applies box blur filter
void box_blur(const uchar* in, uchar* out, int rows, int cols, int window_size);
// creates disparity map
cv::Mat make_disparity_map(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                           const cv::Mat& right_mean, int window_size, int d_first, int d_last);
// cross-checks 2 disparity maps
void cross_check_disparity(cv::Mat& l2r, const cv::Mat& r2l, int disparity);
// fills occlusions in disparity map inplace
void fill_occlusions_disparity(cv::Mat& data, int k_size, int disparity);

// List of kernels:

// returns box blur value for pixel
uchar _kernel_box_blur(const uchar* in, int idx_i, int idx_j, int rows, int cols, int k_size);
// returns zncc value for given (idx_i, idx_j, d)
double _kernel_zncc(const uchar* left, uchar l_mean, const uchar* right, uchar r_mean, int rows,
                    int cols, int k_size, int idx_i, int idx_j, int d);
// fill occlusions in a row
void _kernel_fill_occlusions_disparity(uchar* data, int idx_i, int cols, int k_size, int disparity);

std::pair<cv::Mat, cv::Mat> stereo_compute_disparities_impl(const cv::Mat& left,
                                                            const cv::Mat& right, int window_size,
                                                            int disparity) {
    // const int window_size = 5;  // TODO: this parameter must be optimized

    // 1. find mean images:
    cv::Mat left_mean = cv::Mat::zeros(left.size(), left.type());
    cv::Mat right_mean = cv::Mat::zeros(left.size(), left.type());
    const int rows = left.rows, cols = left.cols;
    box_blur(left.data, left_mean.data, rows, cols, window_size);
    box_blur(right.data, right_mean.data, rows, cols, window_size);

    // 2. find disparity maps (L2R and R2L):
    auto map_l2r =
        make_disparity_map(left, left_mean, right, right_mean, window_size, -disparity, 0);
    auto map_r2l =
        make_disparity_map(right, right_mean, left, left_mean, window_size, 0, disparity);

    return {map_l2r, map_r2l};
}

cv::Mat stereo_compute_disparity(const cv::Mat& left, const cv::Mat& right, int disparity) {
    // sanity checks:
    REQUIRE(left.type() == CV_8UC1);
    REQUIRE(left.type() == right.type());
    REQUIRE(left.dims == 2);
    REQUIRE(left.rows == right.rows);
    REQUIRE(left.cols == right.cols);
    REQUIRE(disparity <= std::numeric_limits<uchar>::max());

    // 1. find disparity maps (L2R and R2L):
    cv::Mat map_l2r, map_r2l;
    std::tie(map_l2r, map_r2l) =
        stereo_compute_disparities_impl(left, right, stereo_common::WINDOW_SIZE, disparity);

    // 2. post process:
    cross_check_disparity(map_l2r, map_r2l, disparity);
    fill_occlusions_disparity(map_l2r, stereo_common::WINDOW_SIZE, disparity);

    return map_l2r;
}

// makes disparity map
cv::Mat make_disparity_map(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                           const cv::Mat& right_mean, int window_size, int d_first, int d_last) {
    const int rows = left.rows, cols = left.cols;

    cv::Mat disparity_map = cv::Mat::zeros(left.size(), CV_8UC1);

    int best_disparity = 0;
    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {

            // find max zncc and corresponding disparity for current pixel:
            double max_zncc = -1.0;  // zncc in range [-1, 1]

            uchar l_mean = left_mean.data[idx_i * cols + idx_j];
            for (int d = d_first; d <= d_last; ++d) {
                uchar r_mean = right_mean.data[idx_i * cols + fix(idx_j + d, cols - 1)];
                double v = _kernel_zncc(left.data, l_mean, right.data, r_mean, rows, cols,
                                        window_size, idx_i, idx_j, d);
                if (max_zncc < v) {
                    max_zncc = v;
                    best_disparity = d;
                }
                if (max_zncc >= stereo_common::ZNCC_THRESHOLD) {
                    break;
                }
            }

            // store absolute value of disparity
            disparity_map.at<uchar>(idx_i, idx_j) = std::abs(best_disparity);
        }
    }

    return disparity_map;
}

// box blur:
void box_blur(const uchar* in, uchar* out, int rows, int cols, int k_size) {
    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {
            out[idx_i * cols + idx_j] = _kernel_box_blur(in, idx_i, idx_j, rows, cols, k_size);
        }
    }
}

void cross_check_disparity(cv::Mat& l2r, const cv::Mat& r2l, int disparity) {
    const int threshold = disparity / 4;  // TODO: this parameter must be optimized
    const int rows = l2r.rows, cols = r2l.cols;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int l2r_pixel = l2r.at<uchar>(i, j);
            int r2l_pixel = r2l.at<uchar>(i, j);

            if (std::abs(l2r_pixel - r2l_pixel) > threshold) {
                l2r.at<uchar>(i, j) = std::min(l2r_pixel, r2l_pixel);
            }
        }
    }
}

void fill_occlusions_disparity(cv::Mat& data, int k_size, int disparity) {
    const int rows = data.rows, cols = data.cols;
    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        // TODO: any better strategy? (e.g. run only for known indices, do not go through the
        //       whole matrix)
        _kernel_fill_occlusions_disparity(data.data, idx_i, cols, k_size, disparity);
    }
}

uchar _kernel_box_blur(const uchar* in, int idx_i, int idx_j, int rows, int cols, int k_size) {
    const double multiplier = 1.0 / (k_size * k_size);
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    for (int i = 0; i < k_size; ++i) {
        const int ii = fix(idx_i + i - center_shift, rows - 1);
        for (int j = 0; j < k_size; ++j) {
            const int jj = fix(idx_j + j - center_shift, cols - 1);
            sum += in[ii * cols + jj];
        }
    }
    return uchar(std::round(multiplier * sum));
}

double _kernel_zncc(const uchar* left, uchar l_mean, const uchar* right, uchar r_mean, int rows,
                    int cols, int k_size, int idx_i, int idx_j, int d) {
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    double std_left = 0.0, std_right = 0.0;

    // TODO: look at cv::integral for speed-up

    for (int i = 0; i < k_size; ++i) {
        const int ii = fix(idx_i + i - center_shift, rows - 1);

        for (int j = 0; j < k_size; ++j) {
            const int jj1 = fix(idx_j + j - center_shift, cols - 1);
            const int jj2 = fix(idx_j + j - center_shift + d, cols - 1);

            const int left_pixel = int(left[ii * cols + jj1]) - int(l_mean);
            const int right_pixel = int(right[ii * cols + jj2]) - int(r_mean);

            sum += left_pixel * right_pixel;

            std_left += std::pow(left_pixel, 2);
            std_right += std::pow(right_pixel, 2);
        }
    }

    // ensure STD DEV >= EPS (otherwise we get Inf)
    std_left = std::max(std::sqrt(std_left), stereo_common::EPS);
    std_right = std::max(std::sqrt(std_right), stereo_common::EPS);

    return double(sum) / (std_left * std_right);
}

void _kernel_fill_occlusions_disparity(uchar* data, int idx_i, int cols, int k_size,
                                       int disparity) {
    // just pick closest non-zero value along current row

    // TODO: how to gracefully handle [disparity, cols - disparity) interval??
    uchar nearest_intensity = 0;
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        uchar pixel = data[idx_i * cols + idx_j];
        if (pixel == stereo_common::UNKNOWN_DISPARITY) {
            data[idx_i * cols + idx_j] = nearest_intensity;
        } else {
            nearest_intensity = pixel;
        }
    }
}

// copy row
void _kernel_copy(const uchar* in, uchar* out, int cols) {
    for (int l = 0; l < cols;) {
        for (; l <= cols - UINT8_NLANES; l += UINT8_NLANES) {
            cv::v_uint8x16 p = cv::v_load(&in[l]);
            cv::v_store(&out[l], p);
        }

        // tail:
        if (l < cols) {
            l = cols - UINT8_NLANES;
        }
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
