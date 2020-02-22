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

constexpr const int MAX_WINDOW_SIZE = 15;
constexpr const double EPS = 0.0005;
constexpr const uchar UNKNOWN_DISPARITY = 0;
constexpr const double ZNCC_THR = 0.995;
}  // namespace

// applies box blur filter
void box_blur(const uchar* in, uchar* out, int rows, int cols, int window_size);

// returns zncc value for given (idx_i, idx_j, d)
double _kernel_zncc(const uchar* left, uchar l_mean, const uchar* right, uchar r_mean, int rows,
                    int cols, int k_size, int idx_i, int idx_j, int disparity);

// creates disparity map
template<typename DisparityCorrector>
cv::Mat make_disparity_map(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                           const cv::Mat& right_mean, int window_size, int disparity,
                           DisparityCorrector f);
cv::Mat make_disparity_map_l2r(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                               const cv::Mat& right_mean, int window_size, int disparity);
cv::Mat make_disparity_map_r2l(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                               const cv::Mat& right_mean, int window_size, int disparity);

// cross-checks 2 disparity maps
cv::Mat cross_check_disparity(cv::Mat& l2r, const cv::Mat& r2l);

// fills occlusions in disparity map inplace
void fill_occlusions_disparity(cv::Mat& data, int k_size);
void _kernel_fill_occlusions_disparity(uchar* data, int idx_i, int idx_j, int rows, int cols,
                                       int k_size);

// postprocesses left disparity map (inplace update). very bad monolithic function but it's faster
void postprocess(cv::Mat& l2r, const cv::Mat& r2l, int k_size, int disparity);

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
        make_disparity_map_l2r(left, left_mean, right, right_mean, window_size, disparity);
    auto map_r2l =
        make_disparity_map_r2l(right, right_mean, left, left_mean, window_size, disparity);

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
    // cv::Mat final = cross_check_disparity(map_l2r, map_r2l);
    // fill_occlusions_disparity(final, window_size);
    postprocess(map_l2r, map_r2l, window_size, disparity);

    return map_l2r;
}

// makes disparity map
cv::Mat make_disparity_map_l2r(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                               const cv::Mat& right_mean, int window_size, int disparity) {
    auto f = [](int best, int) { return best; };
    return make_disparity_map(left, left_mean, right, right_mean, window_size, disparity, f);
}
cv::Mat make_disparity_map_r2l(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                               const cv::Mat& right_mean, int window_size, int disparity) {
    auto f = [](int best, int cols) { return cols - best; };
    return make_disparity_map(left, left_mean, right, right_mean, window_size, disparity, f);
}

template<typename DisparityCorrector>
cv::Mat make_disparity_map(const cv::Mat& left, const cv::Mat& left_mean, const cv::Mat& right,
                           const cv::Mat& right_mean, int window_size, int disparity,
                           DisparityCorrector f) {
    REQUIRE(disparity <= std::numeric_limits<uchar>::max());
    const int rows = left.rows, cols = left.cols;

    cv::Mat disparity_map = cv::Mat::zeros(left.size(), CV_8UC1);

    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {

            // find max zncc and corresponding disparity for current pixel:
            double max_zncc = -2.0;  // zncc in range [-1, 1]
            int best_disparity = 0;

            uchar l_mean = left_mean.data[idx_i * cols + idx_j];
            uchar r_mean = right_mean.data[idx_i * cols + idx_j];

            for (int d = 0; d <= disparity; ++d) {
                double v = _kernel_zncc(left.data, l_mean, right.data, r_mean, rows, cols,
                                        window_size, idx_i, idx_j, d);
                if (max_zncc < v) {
                    max_zncc = v;
                    best_disparity = d;
                }
            }

            // shift range to [0, disparity * 2]
            disparity_map.at<uchar>(idx_i, idx_j) = f(best_disparity, cols);
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
double _kernel_zncc(const uchar* left, uchar l_mean, const uchar* right, uchar r_mean, int rows,
                    int cols, int k_size, int idx_i, int idx_j, int disparity) {
    const int center_shift = (k_size - 1) / 2;

    int64_t sum = 0;  // can get bigger than uchar in theory, so using uint
    double std_left = 0.0, std_right = 0.0;

    // TODO: look at cv::integral for speed-up

    for (int i = 0; i < k_size; ++i) {
        const int ii1 = fix(idx_i + i - center_shift, rows - 1);
        const int ii2 = fix(idx_i + i - center_shift - disparity, rows - 1);

        for (int j = 0; j < k_size; ++j) {
            const int jj = fix(idx_j + j - center_shift, cols - 1);

            int left_pixel = left[ii1 * cols + jj] - l_mean;
            int right_pixel = right[ii2 * cols + jj] - r_mean;

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

void postprocess(cv::Mat& l2r, const cv::Mat& r2l, int k_size, int disparity) {
    const int threshold = disparity / 3;  // TODO: this parameter must be optimized
    const int rows = l2r.rows, cols = r2l.cols;

    // cv::Mat out(l2r.size(), CV_8UC1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int l2r_pixel = l2r.at<uchar>(i, j);
            int r2l_pixel = r2l.at<uchar>(i, j);

            // expected: l2r_pixel = d1, r2l_pixel = -d2 (same point but different sign), thus:
            // if (l2r_pixel + r2l_pixel <==> d1 - d2) > threshold - fill occlusion
            if (std::abs(l2r_pixel + r2l_pixel) > threshold) {
                _kernel_fill_occlusions_disparity(l2r.data, i, j, rows, cols, k_size);
            }
        }
    }
}

cv::Mat cross_check_disparity(cv::Mat& l2r, const cv::Mat& r2l) {
    const int threshold = 10;  // TODO: this parameter must be optimized
    const int rows = l2r.rows, cols = r2l.cols;

    cv::Mat cross_checked = l2r;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int l2r_pixel = l2r.at<uchar>(i, j);
            int r2l_pixel = r2l.at<uchar>(i, j);

            if (std::abs(l2r_pixel - r2l_pixel) > threshold) {
                cross_checked.at<uchar>(i, j) = UNKNOWN_DISPARITY;
            }
        }
    }

    return cross_checked;
}

void fill_occlusions_disparity(cv::Mat& data, int k_size) {
    const int rows = data.rows, cols = data.cols;
    for (int idx_i = 0; idx_i < rows; ++idx_i) {
        for (int idx_j = 0; idx_j < cols; ++idx_j) {
            // TODO: any better strategy? (e.g. run only for known indices, do not go through the
            //       whole matrix)
            if (data.data[idx_i * cols + idx_j] == UNKNOWN_DISPARITY) {
                _kernel_fill_occlusions_disparity(data.data, idx_i, idx_j, rows, cols, k_size);
            }
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

    for (int i = 0; i < k_size; ++i) {
        for (int j = 0; j < k_size; ++j) {
            uchar pixel = data[iis[i] * cols + jjs[j]];
            if (pixel != UNKNOWN_DISPARITY) {
                data[idx_i * cols + idx_j] = pixel;
                return;
            }
        }
    }
}
