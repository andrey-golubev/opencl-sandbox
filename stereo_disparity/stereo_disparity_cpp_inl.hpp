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

const int MAX_WINDOW_SIZE = 11;
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
void fill_occlusions_disparity(cv::Mat& data, int k_size);
void _kernel_fill_occlusions_disparity(uchar* data, int idx_i, int idx_j, int rows, int cols,
                                       int k_size);

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
    REQUIRE(window_size <= MAX_WINDOW_SIZE);

    // 1. find disparity maps (L2R and R2L):
    cv::Mat map_l2r, map_r2l;
    std::tie(map_l2r, map_r2l) =
        stereo_compute_disparities_impl(left, right, window_size, disparity);

    // 2. post process:
    cv::Mat final = cross_check_disparity(map_l2r, map_r2l);
    fill_occlusions_disparity(final, window_size);
    return final;
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

void box_blur(const uchar* in, uchar* out, int rows, int cols, int k_size) {
    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {
            out[idx_i * cols + idx_j] = _kernel_box_blur(in, idx_i, idx_j, rows, cols, k_size);
        }
    }
}

// kernel impl:
double zncc(const uchar* left, const uchar* left_mean, const uchar* right, const uchar* right_mean,
            int rows, int cols, int k_size, int idx_i, int idx_j, int disparity) {
    const int center_shift = (k_size - 1) / 2;

    uint sum = 0;  // can get bigger than uchar in theory, so using uint
    double std_left = 0.0, std_right = 0.0;

#if 0
    uchar l_mean = _kernel_box_blur(left, idx_i, idx_j, rows, cols, k_size);
    uchar r_mean = _kernel_box_blur(right, idx_i, idx_j, rows, cols, k_size);
#else
    uchar l_mean = left_mean[idx_i * cols + idx_j];
    uchar r_mean = right_mean[idx_i * cols + idx_j];
#endif

    for (int i = 0; i < k_size; ++i) {
        const int ii1 = fix(idx_i + i - center_shift, rows - 1);
        const int ii2 = fix(idx_i + i - center_shift - disparity, rows - 1);

        for (int j = 0; j < k_size; ++j) {
            const int jj = fix(idx_j + j - center_shift, cols - 1);

            uint left_pixel = left[ii1 * cols + jj] - l_mean;
            uint right_pixel = right[ii2 * cols + jj] - r_mean;

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

    cv::Mat cross_checked = l2r;

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

void fill_occlusions_disparity(cv::Mat& data, int k_size) {
    const int rows = data.rows, cols = data.cols;
    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {
            _kernel_fill_occlusions_disparity(data.data, idx_i, idx_j, rows, cols, k_size);
        }
    }
}

void _kernel_fill_occlusions_disparity(uchar* data, int idx_i, int idx_j, int rows, int cols,
                                       int k_size) {
    // just pick closest non-zero value:

    const int center_shift = (k_size - 1) / 2;

    // pre-find correct indices
    int iis[MAX_WINDOW_SIZE] = {idx_i};
    int jjs[MAX_WINDOW_SIZE] = {idx_j};

    // equivalent to: for k in {+1, -1, +2, -2, ..., +center_shift, -center_shift}
    for (int k = 1; k <= center_shift; ++k) {
        int i = 2 * k;
        {
            iis[i - 1] = fix(idx_i + k, rows - 1);
            jjs[i - 1] = fix(idx_j + k, cols - 1);
        }
        {
            iis[i - 0] = fix(idx_i - k, rows - 1);
            jjs[i - 0] = fix(idx_j - k, cols - 1);
        }
    }

    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {
            for (int i = 0; i < k_size; ++i) {
                for (int j = 0; j < k_size; ++j) {
                    uchar pixel = data[iis[i] * cols + jjs[j]];
                    if (pixel != 0) {
                        data[idx_i * cols + idx_j] = pixel;
                        return;
                    }
                }
            }
        }
    }
}
