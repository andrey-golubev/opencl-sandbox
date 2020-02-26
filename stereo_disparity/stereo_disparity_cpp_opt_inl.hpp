#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <any>

#include <opencv2/core.hpp>
#include <opencv2/core/hal/intrin.hpp>

#include "common/utils.hpp"

#include "stereo_common.hpp"
#include "stereo_detail.hpp"

#ifndef CV_SIMD
#error "OpenCV's CV_SIMD undefined, impossible to use this header."
#endif

namespace stereo_cpp_opt {

// interface:
cv::Mat stereo_compute_disparity(const cv::Mat& left, const cv::Mat& right, int disparity);

int fix(int v, int max_v) {
    if (v < 0) {
        return -v;
    } else if (v > max_v) {
        const int diff = (v - max_v);
        return max_v - diff;
    }
    return v;
}

int fix(int v, int min_v, int max_v) {
    if (v < min_v) {
        const int diff = (min_v - v);
        return min_v + diff;
    } else if (v > max_v) {
        const int diff = (v - max_v);
        return max_v - diff;
    }
    return v;
}

constexpr const int UINT8_NLANES = cv::v_uint8::nlanes;  // SIMD specific constant

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

void copy_make_border(cv::Mat& out, const cv::Mat& in, int row_border, int col_border) {
    REQUIRE(in.type() == CV_8UC1);
    const int rows = in.rows, cols = in.cols;
    REQUIRE(rows > row_border);
    REQUIRE(cols > col_border);
    REQUIRE(cols >= UINT8_NLANES);  // FIXME?
    REQUIRE(out.rows == rows + row_border * 2);
    REQUIRE(out.cols == cols + col_border * 2);

    // copy input rows
    for (int i = 0; i < rows; ++i) {
        const uchar* in_row = (in.data + i * cols);
        uchar* out_row = (out.data + (i + row_border) * (cols + col_border * 2));
        _kernel_copy_make_border(in_row, out_row, cols, col_border);
    }

    // copy reflected rows (reflect101)
    for (int j = 1; j <= row_border; ++j) {
        // top
        {
            const uchar* in_row = (in.data + j * cols);
            uchar* out_row = (out.data + (row_border - j) * (cols + col_border * 2));
            _kernel_copy_make_border(in_row, out_row, cols, col_border);
        }
        // bottom
        {
            const uchar* in_row = (in.data + (rows - 1 - j) * cols);
            uchar* out_row = (out.data + (rows - 1 + row_border + j) * (cols + col_border * 2));
            _kernel_copy_make_border(in_row, out_row, cols, col_border);
        }
    }
}

cv::Mat copy_make_border(const cv::Mat& in, int row_border, int col_border) {
    REQUIRE(in.type() == CV_8UC1);
    const int rows = in.rows, cols = in.cols;
    REQUIRE(rows > row_border);
    REQUIRE(cols > col_border);
    REQUIRE(cols >= UINT8_NLANES);  // FIXME?

    cv::Mat out = cv::Mat::zeros(cv::Size(cols + col_border * 2, rows + row_border * 2), in.type());
    copy_make_border(out, in, row_border, col_border);
    return out;
}

// idea apply ROI to input, then copy with column border only
void copy_line_border(cv::Mat& out, const cv::Mat& in, detail::HorSlice slice, int col_border) {
    // TOOD: remove requires from this version? (make it unsafe)
    const int cols = in.cols;
    REQUIRE(in.type() == CV_8UC1);
    REQUIRE(slice.y >= 0);
    REQUIRE(slice.y + slice.height <= in.rows);
    REQUIRE(cols > col_border);
    REQUIRE(cols >= UINT8_NLANES);  // FIXME?
    REQUIRE(out.rows == slice.height);
    REQUIRE(out.cols == cols + col_border * 2);

    // copy input rows
    for (int i = slice.y; i < slice.height + slice.y; ++i) {
        const uchar* in_row = (in.data + i * cols);
        uchar* out_row = (out.data + (i - slice.y) * (cols + col_border * 2));
        _kernel_copy_make_border(in_row, out_row, cols, col_border);
    }
}

cv::Mat copy_line_border(const cv::Mat& in, detail::HorSlice slice, int col_border) {
    REQUIRE(in.type() == CV_8UC1);
    const int cols = in.cols;
    REQUIRE(slice.y >= 0);
    REQUIRE(slice.y + slice.height <= in.rows);
    REQUIRE(cols > col_border);
    REQUIRE(cols >= UINT8_NLANES);  // FIXME?

    cv::Mat out = cv::Mat::zeros(cv::Size(cols + col_border * 2, slice.height), in.type());

    copy_line_border(out, in, slice, col_border);

    return out;
}

cv::Mat copy_make_border(const cv::Mat& in, detail::Border b) {
    return copy_make_border(in, b.row_border, b.col_border);
}

uchar _kernel_box_blur(const uchar* in[], int idx_j, int k_size) {
    const double multiplier = 1.0 / (k_size * k_size);
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    for (int i = 0; i < k_size; ++i) {
        for (int j = 0; j < k_size; ++j) {
            const int jj = idx_j + j - center_shift;
            sum += in[i][jj];
        }
    }
    return uchar(std::round(multiplier * sum));
}

// box blur:
void box_blur(uchar* out, const detail::DataView& in_view, int cols, int k_size) {
    const int center_shift = (k_size - 1) / 2;
    const uchar* in[stereo_common::MAX_WINDOW] = {};
    for (int i = 0; i < k_size; ++i) {
        in[i] = in_view.line(i - center_shift);
    }

    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        out[idx_j] = _kernel_box_blur(in, idx_j, k_size);
    }
}

double _kernel_zncc(const uchar* left[], uchar l_mean, const uchar* right[], uchar r_mean,
                    int idx_j, int k_size, int d) {
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    double std_left = 0.0, std_right = 0.0;

    for (int i = 0; i < k_size; ++i) {
        for (int j = 0; j < k_size; ++j) {
            const int jj1 = idx_j + j - center_shift;
            const int jj2 = idx_j + j - center_shift + d;

            const int left_pixel = int(left[i][jj1]) - int(l_mean);
            const int right_pixel = int(right[i][jj2]) - int(r_mean);

            sum += left_pixel * right_pixel;

            std_left += left_pixel * left_pixel;
            std_right += right_pixel * right_pixel;
        }
    }

    // ensure STD DEV >= EPS (otherwise we get Inf)
    std_left = std::max(std::sqrt(std_left), stereo_common::EPS);
    std_right = std::max(std::sqrt(std_right), stereo_common::EPS);

    return double(sum) / (std_left * std_right);
}

// makes disparity map
void make_disparity_map(uchar* out, const detail::DataView& left_view,
                        const detail::DataView& left_mean_view, const detail::DataView& right_view,
                        const detail::DataView& right_mean_view, int cols, int k_size, int d_first,
                        int d_last) {
    // always pick current line for means
    const uchar* left_mean = left_mean_view.line(0);
    const uchar* right_mean = right_mean_view.line(0);

    // prepare lines
    const int center_shift = (k_size - 1) / 2;
    const uchar* left[stereo_common::MAX_WINDOW] = {};
    const uchar* right[stereo_common::MAX_WINDOW] = {};
    for (int i = 0; i < k_size; ++i) {
        left[i] = left_view.line(i - center_shift);
        right[i] = right_view.line(i - center_shift);
    }

    // run zncc for current line
    int best_disparity = 0;
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        // find max zncc and corresponding disparity for current pixel:
        double max_zncc = -1.0;  // zncc in range [-1, 1]

        uchar l_mean = left_mean[idx_j];
        for (int d = d_first; d <= d_last; ++d) {
            uchar r_mean = right_mean[idx_j + d];

            double v = _kernel_zncc(left, l_mean, right, r_mean, idx_j, k_size, d);
            if (max_zncc < v) {
                max_zncc = v;
                best_disparity = d;
            }
            if (max_zncc >= stereo_common::ZNCC_THRESHOLD) {
                break;
            }
        }

        // store absolute value of disparity
        out[idx_j] = std::abs(best_disparity);
    }
}

// TODO: debug abs and min

// pseudo-abs with threshold applied
inline int no_if_abs_thr(int a, int b, int thr) {
    int v = a - b;
    // |value| > thr yields:
    // value > thr OR value < -thr
    return v > thr || v < -thr;
}

inline int no_if_min(int a, int b) {
    int k = (a >= b);
    return k * b + (1 - k) * a;
}

void cross_check_disparity(uchar* out, const detail::DataView& l2r_view,
                           const detail::DataView& r2l_view, int cols, int disparity) {
    const int threshold = disparity / 4;  // TODO: this parameter must be optimized

    // always get the first line
    const uchar* l2r = l2r_view.line(0);
    const uchar* r2l = r2l_view.line(0);

    for (int j = 0; j < cols; ++j) {
        const int l2r_pixel = l2r[j];
        const int r2l_pixel = r2l[j];

        if (std::abs(l2r_pixel - r2l_pixel) > threshold) {
            out[j] = std::min(l2r_pixel, r2l_pixel);
        } else {
            out[j] = l2r_pixel;
        }
    }
}

void fill_occlusions_disparity(uchar* out, const detail::DataView& in_view, int cols) {
    // just pick closest non-zero value along current row

    // always get the first line
    const uchar* in = in_view.line(0);

    // TODO: how to gracefully handle [disparity, cols - disparity) interval??
    uchar nearest_intensity = 0;
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        uchar pixel = in[idx_j];
        if (pixel == stereo_common::UNKNOWN_DISPARITY) {
            out[idx_j] = nearest_intensity;
        } else {
            out[idx_j] = pixel;
            nearest_intensity = pixel;
        }
    }
}

cv::Mat stereo_compute_disparity(const cv::Mat& left, const cv::Mat& right, int disparity) {
    // sanity checks:
    REQUIRE(left.type() == CV_8UC1);
    REQUIRE(left.type() == right.type());
    REQUIRE(left.dims == 2);
    REQUIRE(left.rows == right.rows);
    REQUIRE(left.cols == right.cols);
    REQUIRE(disparity <= std::numeric_limits<uchar>::max());
    REQUIRE(left.rows > stereo_common::MAX_BORDER);
    REQUIRE(left.cols > disparity);

    // many constants here
    const int rows = left.rows;
    const int cols = left.cols;
    constexpr const int border = stereo_common::MAX_BORDER;
    constexpr const detail::Border null_border{0, 0};
    constexpr const detail::Border default_border{border, border};
    const detail::Border disparity_border{border, border + disparity};
    constexpr const int lpi = 1;

    // define all kernels
    auto blur = detail::Kernel<decltype(box_blur)>(box_blur, {default_border});
    auto get_disp = detail::Kernel<decltype(make_disparity_map)>(
        make_disparity_map, {disparity_border, {0, disparity}, disparity_border, {0, disparity}});
    auto cross_check = detail::Kernel<decltype(cross_check_disparity)>(cross_check_disparity,
                                                                       {null_border, null_border});
    auto fill_occlusions = detail::Kernel<decltype(fill_occlusions_disparity)>(
        fill_occlusions_disparity, {null_border});

    // sanity border checks, can be removed later
    REQUIRE(get_disp.border(0) == get_disp.border(2) && get_disp.border(1) == get_disp.border(3));
    REQUIRE(cross_check.border(0) == null_border && cross_check.border(1) == null_border);
    REQUIRE(fill_occlusions.border(0) == null_border);

    // extend inputs with row borders
    cv::Mat in_left = copy_make_border(left, border, 0);
    cv::Mat in_right = copy_make_border(right, border, 0);

    // run pipeline
    cv::Mat out = cv::Mat::zeros(left.size(), CV_8UC1);

    for (int r = 0; r < rows; r += lpi) {
        // a. get mean images:
        cv::Mat left_mean, right_mean;  // outputs
        {
            // 1. extend inputs with borders
            auto blur_border = blur.border(0);
            detail::HorSlice slice{r, lpi + border * 2};
            cv::Mat bordered_left = copy_line_border(in_left, slice, blur_border.col_border);
            cv::Mat bordered_right = copy_line_border(in_right, slice, blur_border.col_border);

            // 2. create input views
            detail::DataView left_view(bordered_left, blur_border);
            detail::DataView right_view(bordered_right, blur_border);

            // 3. create outputs
            left_mean = cv::Mat::zeros(cv::Size(cols, lpi), CV_8UC1);
            right_mean = cv::Mat::zeros(cv::Size(cols, lpi), CV_8UC1);

            // 4. run kernels
            for (int line = 0; line < lpi; ++line) {
                left_view.adjust(line);   // adjust to current line
                right_view.adjust(line);  // adjust to current line

                blur(left_mean.data + line * cols, left_view, cols, stereo_common::MAX_WINDOW);

                blur(right_mean.data + line * cols, right_view, cols, stereo_common::MAX_WINDOW);
            }
        }

        // b. get disparity maps
        cv::Mat map_l2r, map_r2l;
        {
            // 1. extend inputs with borders
            detail::HorSlice slice{r, lpi + border * 2};
            cv::Mat bordered_left = copy_line_border(in_left, slice, get_disp.border(0).col_border);
            cv::Mat bordered_left_mean = copy_make_border(left_mean, get_disp.border(1));
            cv::Mat bordered_right =
                copy_line_border(in_right, slice, get_disp.border(2).col_border);
            cv::Mat bordered_right_mean = copy_make_border(right_mean, get_disp.border(3));

            // 2. create input views
            detail::DataView left_view(bordered_left, get_disp.border(0));
            detail::DataView left_mean_view(bordered_left_mean, get_disp.border(1));
            detail::DataView right_view(bordered_right, get_disp.border(2));
            detail::DataView right_mean_view(bordered_right_mean, get_disp.border(3));

            // 3. create outputs
            map_l2r = cv::Mat::zeros(cv::Size(cols, lpi), CV_8UC1);
            map_r2l = cv::Mat::zeros(cv::Size(cols, lpi), CV_8UC1);

            // 4. run kernels
            for (int line = 0; line < lpi; ++line) {
                left_view.adjust(line);        // adjust to current line
                left_mean_view.adjust(line);   // adjust to current line
                right_view.adjust(line);       // adjust to current line
                right_mean_view.adjust(line);  // adjust to current line

                get_disp(map_l2r.data + line * cols, left_view, left_mean_view, right_view,
                         right_mean_view, cols, stereo_common::MAX_WINDOW, -disparity, 0);

                get_disp(map_r2l.data + line * cols, right_view, right_mean_view, left_view,
                         left_mean_view, cols, stereo_common::MAX_WINDOW, 0, disparity);
            }
        }

        // c. cross check
        cv::Mat cross_checked;
        {
            // 1. extend inputs with borders - not needed here, border always {0, 0}
            cv::Mat bordered_map_l2r = map_l2r;
            cv::Mat bordered_map_r2l = map_r2l;

            // 2. create input views
            detail::DataView map_l2r_view(bordered_map_l2r, cross_check.border(0));
            detail::DataView map_r2l_view(bordered_map_r2l, cross_check.border(1));

            // 3. create outputs
            cross_checked = cv::Mat::zeros(cv::Size(cols, lpi), CV_8UC1);

            // 4. run kernels
            for (int line = 0; line < lpi; ++line) {
                map_l2r_view.adjust(line);  // adjust to current line
                map_r2l_view.adjust(line);  // adjust to current line

                cross_check(cross_checked.data + line * cols, map_l2r_view, map_r2l_view, cols,
                            disparity);
            }
        }

        // d. fill occlusions
        cv::Mat filled;
        {
            // 1. extend inputs with borders - not needed here, border always {0, 0}
            cv::Mat bordered_cross_checked = cross_checked;

            // 2. create input views
            detail::DataView cross_checked_view(bordered_cross_checked, cross_check.border(0));

            // 3. create outputs - in this case it's just the output
            filled = out;

            for (int line = 0; line < lpi; ++line) {
                cross_checked_view.adjust(line);  // adjust to current line

                fill_occlusions(filled.data + (line + r) * cols, cross_checked_view, cols);
            }
        }
    }

    return out;
}
}  // namespace stereo_cpp_opt
