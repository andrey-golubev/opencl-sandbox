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

constexpr const int UINT8_NLANES = cv::v_uint8::nlanes;  // SIMD specific constant

// copy row
void _kernel_copy(const uchar* in, uchar* out, int cols) {
    int l = 0;
    for (; l <= cols - UINT8_NLANES; l += UINT8_NLANES) {
        cv::v_uint8 p = cv::v_load(&in[l]);
        cv::v_store(&out[l], p);
    }

    // tail:
    if (l < cols) {
        l = cols - UINT8_NLANES;
        cv::v_uint8 p = cv::v_load(&in[l]);
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

void copy(cv::Mat& out, const cv::Mat& in, int row_border, int col_border) {
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
        uchar* out_row = (out.data + (i + row_border) * (cols + col_border * 2)) + col_border;
        _kernel_copy(in_row, out_row, cols);
    }
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

// assume input already bordered by k_size
template<typename InType = uchar, typename KType = uchar, typename OutType = int>
void convolve(OutType* out, const InType* in, int rows, int cols, const KType* kernel, int k_size,
              int step = 1) {
    const int border = (k_size - 1) / 2;

    const KType* k[stereo_common::MAX_WINDOW] = {};
    for (int ki = 0; ki < k_size; ++ki) {
        k[ki] = kernel + ki * k_size;
    }

    const InType* data_start = in + border * (cols + border * 2) + border;

    for (int i = 0, oi = 0; i < rows; i += step, oi++) {

        const InType* lines[stereo_common::MAX_WINDOW] = {};
        for (int ki = 0; ki < k_size; ++ki) {
            lines[ki] = data_start + (i + ki - border) * (cols + border * 2);
        }

        for (int j = 0, oj = 0; j < cols; j += step, oj++) {

            OutType sum = 0;
            for (int ki = 0; ki < k_size; ++ki) {
                for (int kj = 0; kj < k_size; ++kj) {
                    sum += k[ki][kj] * lines[ki][j + kj - border];
                }
            }

            out[oi * cols + oj] = sum;
        }
    }
}

cv::Mat mat_conv(const cv::Mat& in, const cv::Mat& kernel, int step) {
    const int k_size = kernel.rows;
    REQUIRE(k_size == kernel.cols);
    const int border = (k_size - 1) / 2;

    // use zero-pad
    cv::Mat bordered_in =
        cv::Mat::zeros(cv::Size(in.cols + border * 2, in.rows + border * 2), in.type());
    copy(bordered_in, in, border, border);

    cv::Mat out = cv::Mat::zeros(in.size(), CV_32SC1);

    convolve((int*)out.data, bordered_in.data, in.rows, in.cols, kernel.data, k_size, step);

    return out;
}

template<typename> struct vec_utils;
template<> struct vec_utils<double> {
    using v_type = cv::v_float32;
    inline static v_type setzero() { return cv::v_setzero_f32(); }
    inline static v_type cvt(const cv::v_int32& in) { return cv::v_cvt_f32(in); }
};
template<> struct vec_utils<float> : vec_utils<double> {};
template<> struct vec_utils<int> {
    using v_type = cv::v_int32;
    inline static v_type setzero() { return cv::v_setzero_s32(); }
    inline static v_type cvt(const cv::v_int32& in) { return in; }
};

// special case of convolution where we know that input and kernel are single lines of the same size
// Note: output is a single value
template<typename InType, typename KType, typename OutType>
void line_convolve(OutType& out, const InType* in, const KType* kernel, int length) {
    OutType sum(0);
    // rely on compiler's auto-vectorization:
    for (int l = 0; l < length; ++l) {
        sum += in[l] * kernel[l];
    }
    out = sum;
}

// optimized overload of line convolution for known input/kernel types
template<typename OutType>
void line_convolve(OutType& out, const short* in, const short* kernel, int length) {
    constexpr const int int16_nlanes = cv::v_int16::nlanes;
    using vu = vec_utils<OutType>;
    OutType sum = 0;
    int l = 0;
    // vectorized part:
    if (length >= int16_nlanes) {
        auto mul_promote = [](const cv::v_int16& a, const cv::v_int16& b) {
            cv::v_int32 a_lo = cv::v_expand_low(a), a_hi = cv::v_expand_high(a);
            cv::v_int32 b_lo = cv::v_expand_low(b), b_hi = cv::v_expand_high(b);
            return a_lo * b_lo + a_hi * b_hi;
        };

        typename vu::v_type s = vu::setzero();
        for (; l <= length - int16_nlanes; l += int16_nlanes) {
            s += vu::cvt(mul_promote(cv::v_load(&in[l]), cv::v_load(&kernel[l])));
        }
        sum = cv::v_reduce_sum(s);
    }

    // scalar part:
    for (; l < length; ++l) {
        sum += int(in[l]) * int(kernel[l]);
    }

    out = sum;
}

// sets partial matrix required for zncc computation
void _set_partial_matrix(short* out, const uchar* in[], uchar mean, int idx, int k_size) {
    constexpr const int int16_nlanes = cv::v_int16::nlanes;
    const int center_shift = (k_size - 1) / 2;

    // vectorized part:
    if (k_size >= int16_nlanes) {
        // promote type: uchar -> short
        const cv::v_int16 mean_v = cv::v_setall_s16(mean);
        auto store_sub = [&](short* out_line, const uchar* line, int j) {
            uchar in_line[int16_nlanes] = {};
            std::memcpy(in_line, line + (idx + j - center_shift), int16_nlanes);
            // load with expansion: uchar -> short
            cv::v_int16 in_v = cv::v_reinterpret_as_s16(cv::v_load_expand(in_line));
            cv::v_store(&out_line[j], in_v - mean_v);
        };

        // for each line: store the line to out buffer with subtracting mean value
        for (int i = 0; i < k_size; ++i) {
            int l = 0;
            const uchar* in_line = in[i];
            short* out_line = out + i * k_size;
            // for each column:
            for (; l <= k_size - int16_nlanes; l += int16_nlanes) {
                store_sub(out_line, in_line, l);
            }

            // handle tail
            if (l < k_size) {
                l = k_size - int16_nlanes;
                store_sub(out_line, in_line, l);
            }
        }
        return;
    }

    // reference:
    for (int i = 0; i < k_size; ++i) {
        for (int j = 0; j < k_size; ++j) {
            out[i * k_size + j] = short(in[i][idx + j - center_shift]) - short(mean);
        }
    }
}

inline double _compute_std_dev(const short* partial_matrix, int k_size) {
    // compute sum of squares via convolution
    int std_dev = 0;
    line_convolve(std_dev, partial_matrix, partial_matrix, k_size * k_size);  // special case
    // ensure STD DEV >= EPS (otherwise we get Inf)
    return std::max(std::sqrt(static_cast<double>(std_dev)), stereo_common::EPS);
}

inline int _compute_sum(const short* left_matrix, const short* right_matrix, int k_size) {
    // compute sum as convolution of 2 different matrices
    int sum = 0;
    line_convolve(sum, left_matrix, right_matrix, k_size * k_size);  // special case
    return sum;
}

double _best_disparity(const uchar* left[], const uchar* left_mean, const uchar* right[],
                       const uchar* right_mean, int idx_j, int k_size, int d_first, int d_last) {
    // find max zncc and corresponding disparity for current pixel:
    double max_zncc = -1.0;  // zncc in range [-1, 1]
    int best_disparity = 0;

    // prepare left matrix
    uchar l_mean = left_mean[idx_j];
    short left_matrix[stereo_common::MAX_WINDOW * stereo_common::MAX_WINDOW] = {};
    _set_partial_matrix(left_matrix, left, l_mean, idx_j, k_size);
    // compute left std_dev
    double std_left = _compute_std_dev(left_matrix, k_size);

    short right_matrix[stereo_common::MAX_WINDOW * stereo_common::MAX_WINDOW] = {};
    for (int d = d_first; d <= d_last; ++d) {
        // prepare right matrix
        uchar r_mean = right_mean[idx_j + d];
        _set_partial_matrix(right_matrix, right, r_mean, idx_j + d, k_size);
        // compute right std_dev
        double std_right = _compute_std_dev(right_matrix, k_size);

        // compute sum in zncc formula
        int sum = _compute_sum(left_matrix, right_matrix, k_size);

        double v = (double)sum / (std_left * std_right);
        if (max_zncc < v) {
            max_zncc = v;
            best_disparity = d;
        }
    }
    return best_disparity;
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
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        int best_disparity =
            _best_disparity(left, left_mean, right, right_mean, idx_j, k_size, d_first, d_last);
        // store absolute value of disparity
        out[idx_j] = std::abs(best_disparity);
    }
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

cv::Mat _internal_stereo_compute_disparity(cv::Mat& out, const cv::Mat& in_left,
                                           const cv::Mat& in_right, detail::HorSlice global_slice,
                                           int rows, int cols, int disparity) {
    // many constants here
    constexpr const int border = stereo_common::MAX_BORDER;
    constexpr const int window = border * 2 + 1;
    constexpr const detail::Border null_border{0, 0};
    constexpr const detail::Border default_border{border, border};
    const detail::Border disparity_border{border, border + disparity};

    auto find_good_lpi = [](int slice_height) {
        const int lpis[] = {8, 4, 2};
        for (int lpi : lpis) {
            if (slice_height < lpi) {
                continue;
            }
            if (slice_height % lpi == 0) {
                return lpi;
            }
        }
        return 1;
    };
    // address lpi > slice rows
    const int lpi = find_good_lpi(global_slice.height);

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

    // create kernel data - allocate inputs/outputs
    const cv::Size line_size(cols, lpi);
    detail::KernelData blur_data_left(detail::in_sizes(line_size, blur), blur.borders(),
                                      {line_size});
    detail::KernelData blur_data_right(detail::in_sizes(line_size, blur), blur.borders(),
                                       {line_size});
    // can use the same disp_data for 2 kernels, but need 2 outputs now (L2R and R2L)
    detail::KernelData disp_data(detail::in_sizes(line_size, get_disp), get_disp.borders(),
                                 {line_size, line_size});
    // cross check reads disparity maps directly
    detail::KernelData cross_check_data(cross_check.borders(), {line_size});
    // fill occlusions reads cross check and writes to output directly
    detail::KernelData fill_occlusions_data(fill_occlusions.borders(), std::vector<cv::Mat>{out});
    REQUIRE(out.data == fill_occlusions_data.out_data(0));

    // run pipeline only for global slice
    for (int r = global_slice.y; r < global_slice.height + global_slice.y; r += lpi) {
        const detail::HorSlice slice{r, lpi + border * 2};  // input slice to use in this iteration

        // 1. get mean images:
        cv::Mat left_mean, right_mean;  // outputs
        {
            // update inputs
            blur_data_left.update_src(0, [&](cv::Mat& out, detail::Border border) {
                copy_line_border(out, in_left, slice, border.col_border);
            });
            blur_data_right.update_src(0, [&](cv::Mat& out, detail::Border border) {
                copy_line_border(out, in_right, slice, border.col_border);
            });

            // re-assign outputs
            left_mean = blur_data_left.out_mat(0);
            right_mean = blur_data_right.out_mat(0);

            // run kernels
            for (int line = 0; line < lpi; ++line) {
                blur_data_left.adjust(line);  // adjust to current line
                blur(left_mean.data + line * cols, blur_data_left.in_view(0), cols, window);

                blur_data_right.adjust(line);  // adjust to current line
                blur(right_mean.data + line * cols, blur_data_right.in_view(0), cols, window);
            }
        }

        // 2. get disparity maps
        cv::Mat map_l2r, map_r2l;
        {
            // update inputs
            disp_data.update_src<0, 1, 2, 3>(
                [&](cv::Mat& out, detail::Border border) {
                    copy_line_border(out, in_left, slice, border.col_border);
                },
                [&](cv::Mat& out, detail::Border border) {
                    copy_make_border(out, left_mean, border.row_border, border.col_border);
                },
                [&](cv::Mat& out, detail::Border border) {
                    copy_line_border(out, in_right, slice, border.col_border);
                },
                [&](cv::Mat& out, detail::Border border) {
                    copy_make_border(out, right_mean, border.row_border, border.col_border);
                });

            // extract views
            auto& left_view = disp_data.in_view(0);
            auto& left_mean_view = disp_data.in_view(1);
            auto& right_view = disp_data.in_view(2);
            auto& right_mean_view = disp_data.in_view(3);

            // re-assign outputs
            map_l2r = disp_data.out_mat(0);
            map_r2l = disp_data.out_mat(1);

            // run kernels
            for (int line = 0; line < lpi; ++line) {
                disp_data.adjust(line);  // adjust to current line

                get_disp(map_l2r.data + line * cols, left_view, left_mean_view, right_view,
                         right_mean_view, cols, window, -disparity, 0);

                get_disp(map_r2l.data + line * cols, right_view, right_mean_view, left_view,
                         left_mean_view, cols, window, 0, disparity);
            }
        }

        // 3. cross check
        cv::Mat cross_checked;
        {
            // update inputs
            cross_check_data.update_view<0, 1>(
                [&](detail::DataView& view) { view = detail::DataView(map_l2r, null_border); },
                [&](detail::DataView& view) { view = detail::DataView(map_r2l, null_border); });

            // extract views
            auto& map_l2r_view = cross_check_data.in_view(0);
            auto& map_r2l_view = cross_check_data.in_view(1);

            // re-assign outputs
            cross_checked = cross_check_data.out_mat(0);

            // run kernels
            for (int line = 0; line < lpi; ++line) {
                cross_check_data.adjust(line);  // adjust to current line

                cross_check(cross_checked.data + line * cols, map_l2r_view, map_r2l_view, cols,
                            disparity);
            }
        }

        // 4. fill occlusions
        {
            // update inputs
            fill_occlusions_data.update_view(0, [&](detail::DataView& view) {
                view = detail::DataView(cross_checked, null_border);
            });

            // access output
            auto data = fill_occlusions_data.out_data(0);

            // run kernels
            for (int line = 0; line < lpi; ++line) {
                fill_occlusions_data.adjust(line);  // adjust to current line

                fill_occlusions(data + (line + r) * cols, fill_occlusions_data.in_view(0), cols);
            }
        }
    }

    return out;
}

// interface method:
cv::Mat stereo_compute_disparity(const cv::Mat& left, const cv::Mat& right, int disparity) {
    const int rows = left.rows;
    const int cols = left.cols;
    constexpr const int border = stereo_common::MAX_BORDER;

    // sanity checks:
    REQUIRE(left.type() == CV_8UC1);
    REQUIRE(left.type() == right.type());
    REQUIRE(left.dims == 2);
    REQUIRE(left.rows == right.rows);
    REQUIRE(left.cols == right.cols);
    REQUIRE(disparity <= std::numeric_limits<uchar>::max());
    REQUIRE(left.rows > stereo_common::MAX_BORDER);
    REQUIRE(left.cols > disparity + stereo_common::MAX_BORDER);

    // if there are not enough rows < max threads, just parallelize each row
    const int threads = std::min(thr::get_max_threads(), left.rows);

    // prepare inputs and outputs:
    // extend inputs with row borders - required for correct reflection handling
    cv::Mat in_left = copy_make_border(left, border, 0);
    cv::Mat in_right = copy_make_border(right, border, 0);
    // allocate output
    cv::Mat out = cv::Mat::zeros(left.size(), CV_8UC1);

    // parallelize
    thr::parallel_for(threads, [&](int slice_n, int total_slices) {
        int y = 0;

        auto lines_per_thread = rows / total_slices;
        const auto remainder = rows % total_slices;

        if (slice_n < remainder) {
            lines_per_thread++;
            y = slice_n * lines_per_thread;
        } else {
            y = remainder * (lines_per_thread + 1) + (slice_n - remainder) * lines_per_thread;
        }

        if (lines_per_thread <= 0) {
            return;
        }

        // run algorithm for horizontal image patch
        detail::HorSlice patch{y, lines_per_thread};  // process some patch in this thread
        _internal_stereo_compute_disparity(out, in_left, in_right, patch, rows, cols, disparity);
    });

    return out;
}
}  // namespace stereo_cpp_opt
