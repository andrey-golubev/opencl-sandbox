#pragma once

#include <cmath>
#include <utility>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>

#include "common/require.hpp"

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
}  // namespace

// applies box blur filter
void box_blur(const uchar* in, uchar* out, int rows, int cols, int window_size);

// returns zncc value for given (idx_i, idx_j, d)
double zncc(const uchar* left, const uchar* left_mean, const uchar* right, const uchar* right_mean,
            int rows, int cols, int window_size, int idx_i, int idx_j, int disparity);

// creates disparity map
cv::Mat make_disparity_map(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                           const cv::Mat& right_mean, int window_size, int disparity);

// cross-checks 2 disparity maps
cv::Mat cross_check_disparity(const cv::Mat& l2r, cv::Mat& r2l);

// fills occlusions in disparity map inplace
cv::Mat fill_occlusions_disparity(const cv::Mat& mat);

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
    auto map_l2r = make_disparity_map(left, left_mean, right, right_mean, window_size, disparity);
    auto map_r2l = make_disparity_map(right, right_mean, left, left_mean, window_size, disparity);

    return {map_l2r, map_r2l};
}

cv::Mat stereo_compute_disparity(const cv::Mat& left, const cv::Mat& right, int window_size,
                                 int disparity) {
    // sanity checks:
    REQUIRE(left.type() == CV_8UC1);
    REQUIRE(left.type() == right.type());
    REQUIRE(left.dims == 2);
    REQUIRE(left.rows == right.rows);
    REQUIRE(left.cols == right.cols);

    // 1. find disparity maps (L2R and R2L):
    cv::Mat map_l2r, map_r2l;
    std::tie(map_l2r, map_r2l) =
        stereo_compute_disparities_impl(left, right, window_size, disparity);

    // 2. post process:
    return fill_occlusions_disparity(cross_check_disparity(map_l2r, map_r2l));
}

// makes disparity map
cv::Mat make_disparity_map(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                           const cv::Mat& right_mean, int window_size, int disparity) {
    REQUIRE(disparity <= std::numeric_limits<uchar>::max());
    const int rows = left.rows, cols = left.cols;

    cv::Mat disparity_map = cv::Mat::zeros(left.size(), CV_8UC1);

    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {

            // find max zncc and corresponding disparity for current pixel:
            double max_zncc = -2.0;  // zncc in range [-1, 1]
            int best_disparity = 0;  // d in range (0, disparity], use 0 for auto occlusion fill

            for (int d = 0; d <= disparity; ++d) {
                double v = zncc(left.data, left_mean.data, right.data, right_mean.data, rows, cols,
                                window_size, idx_i, idx_j, d);
                if (max_zncc < v) {
                    max_zncc = v;
                    best_disparity = d;
                }
            }

            disparity_map.at<uchar>(idx_i, idx_j) = best_disparity;
        }
    }

    return disparity_map;
}

// box blur:
void box_blur(const uchar* in, uchar* out, int rows, int cols, int k_size) {
    const double multiplier = 1.0 / (k_size * k_size);
    const int center_shift = (k_size - 1) / 2;

    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {
            int sum = 0;
            for (int i = 0; i < k_size; ++i) {
                const int ii = fix(idx_i + i - center_shift, rows - 1);
                for (int j = 0; j < k_size; ++j) {
                    const int jj = fix(idx_j + j - center_shift, cols - 1);
                    sum += in[ii * cols + jj];
                }
            }
            out[idx_i * cols + idx_j] = uchar(std::round(multiplier * sum));
        }
    }
}

// kernel impl:
double zncc(const uchar* left, const uchar* left_mean, const uchar* right, const uchar* right_mean,
            int rows, int cols, int k_size, int idx_i, int idx_j, int disparity) {
    const int center_shift = (k_size - 1) / 2;

    uint sum = 0;  // can get bigger than uchar in theory, so using uint
    double std_left = 0.0, std_right = 0.0;

    for (int i = 0; i < k_size; ++i) {
        const int ii1 = fix(idx_i + i - center_shift, rows - 1);
        const int ii2 = fix(idx_i + i - center_shift - disparity, rows - 1);

        for (int j = 0; j < k_size; ++j) {
            const int jj = fix(idx_j + j - center_shift, cols - 1);

            uint left_pixel = left[ii1 * cols + jj] - left_mean[idx_i * cols + idx_j];
            uint right_pixel = right[ii2 * cols + jj] - right_mean[idx_i * cols + idx_j];

            sum += left_pixel * right_pixel;

            std_left += std::pow(left_pixel, 2);
            std_right += std::pow(right_pixel, 2);
        }
    }

    std_left = std::sqrt(std_left);
    std_right = std::sqrt(std_right);

    return double(sum) / (std_left * std_right);
}

cv::Mat cross_check_disparity(const cv::Mat& l2r, cv::Mat& r2l) {
    const int threshold = 10;  // TODO: this parameter must be optimized
    const int rows = l2r.rows, cols = r2l.cols;

    cv::Mat cross_checked = r2l;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int l2r_pixel = l2r.at<uchar>(i, j);
            int r2l_pixel = r2l.at<uchar>(i, j);

            if (std::abs(l2r_pixel - r2l_pixel) > threshold) {
                cross_checked.at<uchar>(i, j) = 0;
            }
        }
    }

    return cross_checked;
}

cv::Mat fill_occlusions_disparity(const cv::Mat& in) {
    // TODO: implement something
    return in;
}
