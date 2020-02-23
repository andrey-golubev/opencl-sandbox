#pragma once

#include <cmath>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>

#include "common/utils.hpp"

namespace {
int fix(int v, int max_v) {
    if (v < 0) {
        return -v;
    } else if (v > max_v) {
        const int diff = (v - max_v);
        return max_v - diff;
    }
    return v;
}

constexpr const double EPS = 0.0005;
constexpr const uchar UNKNOWN_DISPARITY = 0;
constexpr const int WINDOW_SIZE = 11;

struct Range1d {
    int first = 0;
    int last = 0;
};
}  // namespace

namespace stereo_cpp_base {
// List of functions:
// applies box blur filter
void box_blur(const uchar* in, uchar* out, int rows, int cols, int window_size);
// creates disparity map
cv::Mat make_disparity_map(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                           const cv::Mat& right_mean, int window_size, int disparity, int lds,
                           int rds);
// cross-checks 2 disparity maps
void cross_check_disparity(cv::Mat& l2r, const cv::Mat& r2l, int disparity);
// fills occlusions in disparity map inplace
void fill_occlusions_disparity(cv::Mat& data, int k_size, int disparity);

// List of kernels:
// returns box blur value for pixel
uchar _kernel_box_blur(const uchar* in, int idx_i, int idx_j, int rows, int cols, int k_size);
// returns zncc value for given (idx_i, idx_j, d)
double _kernel_zncc(const uchar* left, uchar l_mean, const uchar* right, uchar r_mean, int rows,
                    int cols, int k_size, int idx_i, int idx_j, int l_shift, int r_shift);
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
    constexpr const int L2R_MAGIC_SHIFT_COEFFS[2] = {0, -1};
    constexpr const int R2L_MAGIC_SHIFT_COEFFS[2] = {1, 0};
    auto map_l2r = make_disparity_map(left, left_mean, right, right_mean, window_size, disparity,
                                      L2R_MAGIC_SHIFT_COEFFS[0], L2R_MAGIC_SHIFT_COEFFS[1]);
    auto map_r2l = make_disparity_map(right, right_mean, left, left_mean, window_size, disparity,
                                      R2L_MAGIC_SHIFT_COEFFS[0], R2L_MAGIC_SHIFT_COEFFS[1]);

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
        stereo_compute_disparities_impl(left, right, WINDOW_SIZE, disparity);

    // 2. post process:
    cross_check_disparity(map_l2r, map_r2l, disparity);
    fill_occlusions_disparity(map_l2r, WINDOW_SIZE, disparity);

    return map_l2r;
}

// makes disparity map
cv::Mat make_disparity_map(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                           const cv::Mat& right_mean, int window_size, int disparity, int lds,
                           int rds) {
    const int rows = left.rows, cols = left.cols;

    cv::Mat disparity_map = cv::Mat::zeros(left.size(), CV_8UC1);

    int best_disparity = 0;
    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {

            // find max zncc and corresponding disparity for current pixel:
            double max_zncc = -1.0;  // zncc in range [-1, 1]

            for (int d = -disparity; d <= disparity; ++d) {
                int l_shift = lds * d;
                int r_shift = rds * d;
                uchar l_mean = left_mean.data[idx_i * cols + fix(idx_j + l_shift, cols - 1)];
                uchar r_mean = right_mean.data[idx_i * cols + fix(idx_j + r_shift, cols - 1)];
                double v = _kernel_zncc(left.data, l_mean, right.data, r_mean, rows, cols,
                                        window_size, idx_i, idx_j, l_shift, r_shift);
                if (max_zncc < v) {
                    max_zncc = v;
                    best_disparity = d;
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
                l2r.at<uchar>(i, j) = UNKNOWN_DISPARITY;
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
                    int cols, int k_size, int idx_i, int idx_j, int l_shift, int r_shift) {
    const int center_shift = (k_size - 1) / 2;

    int64_t sum = 0;  // can get bigger than uchar in theory, so using uint
    double std_left = 0.0, std_right = 0.0;

    // TODO: look at cv::integral for speed-up

    for (int i = 0; i < k_size; ++i) {
        const int ii = fix(idx_i + i - center_shift, rows - 1);

        for (int j = 0; j < k_size; ++j) {
            const int jj1 = fix(idx_j + j - center_shift + l_shift, cols - 1);
            const int jj2 = fix(idx_j + j - center_shift + r_shift, cols - 1);

            int left_pixel = left[ii * cols + jj1] - l_mean;
            int right_pixel = right[ii * cols + jj2] - r_mean;

            sum += int64_t(left_pixel) * right_pixel;

            std_left += std::pow(left_pixel, 2);
            std_right += std::pow(right_pixel, 2);
        }
    }

    // ensure STD DEV >= EPS (otherwise we get Inf)
    std_left = std::max(std::sqrt(std_left), EPS);
    std_right = std::max(std::sqrt(std_right), EPS);

    return double(sum) / (std_left * std_right);
}

void _kernel_fill_occlusions_disparity(uchar* data, int idx_i, int cols, int k_size,
                                       int disparity) {
    // just pick closest non-zero value along current row

    // TODO: how to gracefully handle [disparity, cols - disparity) interval??
    uchar nearest_intensity = 0;
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        uchar pixel = data[idx_i * cols + idx_j];
        if (pixel == UNKNOWN_DISPARITY) {
            data[idx_i * cols + idx_j] = nearest_intensity;
        } else {
            nearest_intensity = pixel;
        }
    }
}
}  // namespace stereo_cpp_base
