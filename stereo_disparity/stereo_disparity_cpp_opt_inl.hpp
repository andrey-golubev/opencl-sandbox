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

cv::Mat copy_make_border(const cv::Mat& in, int row_border, int col_border) {
    REQUIRE(in.type() == CV_8UC1);
    const int rows = in.rows, cols = in.cols;
    REQUIRE(rows > row_border);
    REQUIRE(cols > col_border);
    REQUIRE(cols >= UINT8_NLANES);  // FIXME?

    cv::Mat out = cv::Mat::zeros(cv::Size(cols + col_border * 2, rows + row_border * 2), in.type());

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

    return out;
}

uchar _kernel_box_blur(const uchar* in, int idx_j, int cols, int k_size) {
    const double multiplier = 1.0 / (k_size * k_size);
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    for (int i = 0; i < k_size; ++i) {
        const int ii = i - center_shift;
        for (int j = 0; j < k_size; ++j) {
            const int jj = idx_j + j - center_shift;
            sum += in[ii * cols + jj];
        }
    }
    return uchar(std::round(multiplier * sum));
}

// box blur:
void box_blur(uchar* out, const uchar* in, int cols, int k_size) {
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        out[idx_j] = _kernel_box_blur(in, idx_j, cols, k_size);
    }
}

double _kernel_zncc(const uchar* left, uchar l_mean, const uchar* right, uchar r_mean, int cols,
                    int k_size, int idx_j, int d) {
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    double std_left = 0.0, std_right = 0.0;

    // TODO: look at cv::integral for speed-up

    for (int i = 0; i < k_size; ++i) {
        const int ii = i - center_shift;

        for (int j = 0; j < k_size; ++j) {
            const int jj1 = idx_j + j - center_shift;
            const int jj2 = idx_j + j - center_shift + d;

            const int left_pixel = int(left[ii * cols + jj1]) - int(l_mean);
            const int right_pixel = int(right[ii * cols + jj2]) - int(r_mean);

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
void make_disparity_map(uchar* out, const uchar* left, const uchar* left_mean, const uchar* right,
                        const uchar* right_mean, int cols, int window_size, int d_first,
                        int d_last) {
    int best_disparity = 0;
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        // find max zncc and corresponding disparity for current pixel:
        double max_zncc = -1.0;  // zncc in range [-1, 1]

        uchar l_mean = left_mean[idx_j];
        for (int d = d_first; d <= d_last; ++d) {
            uchar r_mean = right_mean[idx_j + d];
            double v = _kernel_zncc(left, l_mean, right, r_mean, cols, window_size, idx_j, d);
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

void cross_check_disparity(uchar* out, const uchar* l2r, const uchar* r2l, int cols,
                           int disparity) {
    const int threshold = disparity / 4;  // TODO: this parameter must be optimized

    for (int j = 0; j < cols; ++j) {
        int l2r_pixel = l2r[j];
        int r2l_pixel = r2l[j];

        if (std::abs(l2r_pixel - r2l_pixel) > threshold) {
            out[j] = stereo_common::UNKNOWN_DISPARITY;
        } else {
            out[j] = l2r_pixel;
        }
    }
}

void fill_occlusions_disparity(uchar* out, const uchar* in, int cols, int disparity) {
    // just pick closest non-zero value along current row

    // TODO: how to gracefully handle [disparity, cols - disparity) interval??
    uchar nearest_intensity = 0;
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        uchar pixel = in[idx_j];
        if (pixel == stereo_common::UNKNOWN_DISPARITY) {
            out[idx_j] = nearest_intensity;
        } else {
            nearest_intensity = pixel;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// details:

template<typename Signature> struct Kernel;
template<typename R, typename... Args> struct Kernel<R(Args...)> {
    using Invokable = R (*)(Args...);

    Kernel(Invokable f, int border) : m_opaque_function(f), m_border(border) {}
    Kernel(const Kernel&) = default;
    Kernel(Kernel&&) = default;
    Kernel& operator=(const Kernel&) = default;
    Kernel& operator=(Kernel&&) = default;
    ~Kernel() = default;

    R operator()(Args... args) { return m_opaque_function(args...); }
    Invokable& get() { return m_opaque_function; }

    int border() const { return m_border; }
    int max_lines() const { return m_border * 2 + 1; }

private:
    Invokable m_opaque_function;  // opaque function pointer
    int m_border = 0;             // border size for opaque function
};

struct OpaqueKernel {
    template<typename Type>
    OpaqueKernel(Type&& var) : m_holder(std::make_unique<Holder<Type>>(std::forward<Type>(var))) {}

    // TODO: fake copy semantics through move
    OpaqueKernel(const OpaqueKernel& other) : m_holder(other.m_holder->clone()) {}
    OpaqueKernel(OpaqueKernel&& other) : m_holder(std::move(other.m_holder)) {}

    struct Base {
        using Ptr = std::unique_ptr<Base>;
        virtual int border() const = 0;
        virtual int max_lines() const = 0;
        virtual Ptr clone() const = 0;
        virtual ~Base() = default;
    };

    template<typename Type> struct Holder : Base {
        Type m_var;
        Holder(Type var) : m_var(var) {}
        int border() const override { return m_var.border(); }
        int max_lines() const override { return m_var.max_lines(); }
        Base::Ptr clone() const override { return std::make_unique<Holder<Type>>(*this); }
    };

    int border() const { return m_holder->border(); }
    int max_lines() const { return m_holder->max_lines(); }

private:
    typename Base::Ptr m_holder;
};
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename... Kernels> int max_lpi(Kernels... args) {
    std::vector<OpaqueKernel> kernels({args...});
    int max_lines = 0;
    for (const auto& k : kernels) {
        max_lines = std::max(max_lines, k.max_lines());
    }
    return max_lines;
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

    const int rows = left.rows;
    const int cols = left.cols;

#define BORDER 0
    // 1. Define all kernels
#if BORDER
    auto make_border = Kernel<decltype(copy_make_border)>(copy_make_border, 1);
#endif
    // stereo_common::MAX_WINDOW is worst case
    auto blur = Kernel<decltype(box_blur)>(box_blur, stereo_common::MAX_BORDER);
    auto get_disp =
        Kernel<decltype(make_disparity_map)>(make_disparity_map, stereo_common::MAX_BORDER);
    auto cross_check = Kernel<decltype(cross_check_disparity)>(cross_check_disparity, 0);
    auto fill_occlusions =
        Kernel<decltype(fill_occlusions_disparity)>(fill_occlusions_disparity, 0);

    // 2. find out max number of rows
    size_t max_rows = max_lpi(
#if BORDER
        make_border,
#endif
        blur, get_disp, cross_check, fill_occlusions);
    (void)max_rows;

    // 3. run pipeline
    cv::Mat out = cv::Mat::zeros(left.size(), left.type());

    // TODO: on each iteration, do copy_make_border(column_border) [but not rows!]

    // TODO: remove +disparity here, it's awful
    cv::Mat bordered_left =
        copy_make_border(left, stereo_common::MAX_BORDER, stereo_common::MAX_BORDER + disparity);
    cv::Mat bordered_right =
        copy_make_border(right, stereo_common::MAX_BORDER, stereo_common::MAX_BORDER + disparity);
    const int border_shift = stereo_common::MAX_BORDER + disparity;

    // TODO: probably shouldn't go through all the rows here?
    for (int r = 0; r < rows; ++r)
    // for single row...
    {
        // a. get mean images:
        cv::Mat left_mean, right_mean;
        {
            // TODO: real max lines here is multiplication of all successor max_lines()??
            left_mean = cv::Mat::zeros(cv::Size(cols, get_disp.max_lines()), CV_8UC1);
            for (int line = 0; line < get_disp.max_lines(); ++line) {
                const int in_shift = (line + r) * (cols + border_shift) + border_shift;
                blur(left_mean.data + line * cols, bordered_left.data + in_shift, cols,
                     stereo_common::MAX_WINDOW);
            }
            left_mean = copy_make_border(left_mean, 0, disparity);

            right_mean = cv::Mat::zeros(cv::Size(cols, get_disp.max_lines()), CV_8UC1);
            for (int line = 0; line < get_disp.max_lines(); ++line) {
                const int in_shift = (line + r) * (cols + border_shift) + border_shift;
                blur(right_mean.data + line * cols, bordered_right.data + in_shift, cols,
                     stereo_common::MAX_WINDOW);
            }
            right_mean = copy_make_border(right_mean, 0, disparity);
        }

        // b. get disparity maps
        cv::Mat map_l2r, map_r2l;
        {
            map_l2r = cv::Mat::zeros(cv::Size(cols, cross_check.max_lines()), CV_8UC1);
            for (int line = 0; line < cross_check.max_lines(); ++line) {
                const int in_shift = (line + r) * (cols + border_shift) + border_shift;
                const int in_mean_shift = (line + r) * (cols + disparity) + disparity;

                get_disp(map_l2r.data + line * cols, left.data + in_shift,
                         left_mean.data + in_mean_shift, right.data + in_shift,
                         right_mean.data + in_mean_shift, cols, stereo_common::MAX_WINDOW,
                         -disparity, 0);
            }
            // map_l2r = copy_make_border(map_l2r, 0, cross_check.border());

            map_r2l = cv::Mat::zeros(cv::Size(cols, cross_check.max_lines()), CV_8UC1);
            for (int line = 0; line < cross_check.max_lines(); ++line) {
                const int in_shift = (line + r) * (cols + border_shift) + border_shift;
                const int in_mean_shift = (line + r) * (cols + disparity) + disparity;

                get_disp(map_r2l.data + line * cols, right.data + in_shift,
                         right_mean.data + in_mean_shift, left.data + in_shift,
                         left_mean.data + in_mean_shift, cols, stereo_common::MAX_WINDOW, 0,
                         disparity);
            }
            // map_r2l = copy_make_border(map_r2l, 0, cross_check.border());
        }

        // c. cross check
        cv::Mat cross_checked;
        {
            cross_checked = cv::Mat::zeros(cv::Size(cols, fill_occlusions.max_lines()), CV_8UC1);
            for (int line = 0; line < fill_occlusions.max_lines(); ++line) {
                const int in_shift = (line + r) * cols;

                cross_check(cross_checked.data + line * cols, map_l2r.data + in_shift,
                            map_r2l.data + in_shift, cols, disparity);
            }
            // cross_checked = copy_make_border(cross_checked, 0, fill_occlusions.border());
        }

        // d. fill occlusions
        {
            const int in_shift = r * cols;
            fill_occlusions(out.data + r * cols, cross_checked.data + in_shift, cols, disparity);
        }
    }

    return out;
}
}  // namespace stereo_cpp_opt
