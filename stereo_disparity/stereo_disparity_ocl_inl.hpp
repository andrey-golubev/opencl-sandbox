#pragma once

#include <cmath>
#include <regex>
#include <utility>

#include <opencv2/core.hpp>

#include "common/utils.hpp"
#include "common_ocl/executor.hpp"
#include "common_ocl/utils.hpp"
#include "common_ocl/wrappers.hpp"

#include "stereo_common.hpp"

namespace stereo_ocl_base {
// TODO: optimize indices in box_blur and other kernels
std::string program_str(R"(
// helper functions:
int fix(int v, int max_v) {
    if (v < 0) {
        return -v;
    } else if (v > max_v) {
        const int diff = (v - max_v);
        return max_v - diff;
    }
    return v;
}
bool out_of_bounds(int x, int l, int r) {
    if (x < l || x > r) {
        return true;
    }
    return false;
}

// defines:
#define EPS <re:EPS>
#define UNKNOWN_DISPARITY <re:UNKNOWN_DISPARITY>
#define MAX_WINDOW <re:MAX_WINDOW>
#define MAX_BORDER <re:MAX_BORDER>

// kernels and subroutines:
__kernel void box_blur(__global uchar* out, __global const uchar* in, int rows, int cols,
                       int k_size) {
    const int idx_i = get_global_id(0);  // pixel index for row
    const int idx_j = get_global_id(1);  // pixel index for col

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1) || out_of_bounds(idx_j, 0, cols - 1)) {
        return;
    }

    const int center_shift = (k_size - 1) / 2;

    // prepare lines in advance
    __global const uchar* in_lines[MAX_WINDOW] = {};
    for (int i = 0; i < k_size; ++i) {
        const int ii = fix(idx_i + i - center_shift, rows - 1);
        in_lines[i] = in + ii * cols;
    }

    // prepare column indices in advance
    int js[MAX_WINDOW] = {};
    for (int j = 0; j < k_size; ++j) {
        js[j] = fix(idx_j + j - center_shift, cols - 1);
    }

    // main loop
    uint sum = 0;
    for (int i = 0; i < k_size; ++i) {
        for (int j = 0; j < k_size; ++j) {
            sum += in_lines[ i ][ js[j] ];
        }
    }

    const double multiplier = 1.0 / (k_size * k_size);
    out[idx_i * cols + idx_j] = round(multiplier * sum);
}

double zncc(__global const uchar* left, uchar l_mean, __global const uchar* right, uchar r_mean,
            int rows, int cols, int k_size, int idx_i, int idx_j, int d) {
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    double std_left = 0.0, std_right = 0.0;

    // prepare lines in advance
    __global const uchar* left_lines[MAX_WINDOW] = {};
    __global const uchar* right_lines[MAX_WINDOW] = {};
    for (int i = 0; i < k_size; ++i) {
        const int ii = fix(idx_i + i - center_shift, rows - 1);
        left_lines[i] = left + ii * cols;
        right_lines[i] = right + ii * cols;
    }

    // prepare column indices in advance
    int js1[MAX_WINDOW] = {};
    int js2[MAX_WINDOW] = {};
    for (int j = 0; j < k_size; ++j) {
        const int jj = idx_j + j - center_shift;
        js1[j] = fix(jj, cols - 1);
        js2[j] = fix(jj + d, cols - 1);
    }

    // main loop
    for (int i = 0; i < k_size; ++i) {
        for (int j = 0; j < k_size; ++j) {
            const int left_pixel = (int)(left_lines[ i ][ js1[j] ]) - (int)(l_mean);
            const int right_pixel = (int)(right_lines[ i ][ js2[j] ]) - (int)(r_mean);

            sum += left_pixel * right_pixel;

            std_left += pow((double)left_pixel, 2);
            std_right += pow((double)right_pixel, 2);
        }
    }

    // ensure STD DEV >= EPS (otherwise we get Inf)
    std_left = max(sqrt(std_left), EPS);
    std_right = max(sqrt(std_right), EPS);

    const double dsum = sum;
    return dsum / (std_left * std_right);
}

__kernel void make_disparity_map(__global uchar* out, __global const uchar* left,
                                 __global const uchar* left_mean, __global const uchar* right,
                                 __global const uchar* right_mean, int rows, int cols,
                                 int window_size, int d_first, int d_last) {
    const int idx_i = get_global_id(0);  // pixel index for row
    const int idx_j = get_global_id(1);  // pixel index for col

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1) || out_of_bounds(idx_j, 0, cols - 1)) {
        return;
    }

    int best_disparity = 0;
    // find max zncc and corresponding disparity for current pixel:
    double max_zncc = -1.0;  // zncc in range [-1, 1]

    uchar l_mean = left_mean[idx_i * cols + idx_j];
    for (int d = d_first; d <= d_last; ++d) {
        uchar r_mean = right_mean[idx_i * cols + fix(idx_j + d, cols - 1)];
        double v = zncc(left, l_mean, right, r_mean, rows, cols, window_size, idx_i, idx_j, d);
        if (max_zncc < v) {
            max_zncc = v;
            best_disparity = d;
        }
    }

    // store absolute value of disparity
    out[idx_i * cols + idx_j] = abs(best_disparity);
}

__kernel void cross_check_disparity(__global uchar* l2r, __global const uchar* r2l, int rows,
                                    int cols, int disparity) {
    const int idx_i = get_global_id(0);  // pixel index for row
    const int idx_j = get_global_id(1);  // pixel index for col

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1) || out_of_bounds(idx_j, 0, cols - 1)) {
        return;
    }

    const int threshold = disparity / 4;

    const int l2r_pixel = l2r[idx_i * cols + idx_j];
    const int r2l_pixel = r2l[idx_i * cols + idx_j];
    if (abs(l2r_pixel - r2l_pixel) > threshold) {
        l2r[idx_i * cols + idx_j] = min(l2r_pixel, r2l_pixel);
    }
}

__kernel void fill_occlusions_disparity(__global uchar* data, int rows, int cols, int disparity) {
    const int idx_i = get_global_id(0);  // pixel index for row

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1)) {
        return;
    }

    // just pick closest non-zero value along current row

    uchar nearest_intensity = 0;
    for (int idx_j = 0; idx_j < cols; ++idx_j) {
        const uchar pixel = data[idx_i * cols + idx_j];
        if (pixel == UNKNOWN_DISPARITY) {
            data[idx_i * cols + idx_j] = nearest_intensity;
        } else {
            nearest_intensity = pixel;
        }
    }
}
)");

template<typename T> void replace_param(std::string& string, std::string search_for, T param) {
    std::regex re(search_for);
    string = std::regex_replace(string, re, std::to_string(param));
}

void set_parameters(std::string& ocl_program) {
    replace_param(ocl_program, "<re:EPS>", stereo_common::EPS);
    replace_param(ocl_program, "<re:UNKNOWN_DISPARITY>", stereo_common::UNKNOWN_DISPARITY);
    replace_param(ocl_program, "<re:MAX_WINDOW>", stereo_common::MAX_WINDOW);
    replace_param(ocl_program, "<re:MAX_BORDER>", stereo_common::MAX_BORDER);
}

cv::Mat stereo_compute_disparity(const cv::Mat& left, const cv::Mat& right, int disparity,
                                 size_t platform_idx = 0, size_t device_idx = 0) {
    // sanity checks:
    REQUIRE(left.type() == CV_8UC1);
    REQUIRE(left.type() == right.type());
    REQUIRE(left.dims == 2);
    REQUIRE(left.rows == right.rows);
    REQUIRE(left.cols == right.cols);
    REQUIRE(disparity <= std::numeric_limits<uchar>::max());

    set_parameters(program_str);

    static OclExecutor e(
        OclPrimitives(platform_idx, device_idx), program_str,
        {"box_blur", "make_disparity_map", "cross_check_disparity", "fill_occlusions_disparity"});

    const size_t memory_size = left.total() * left.elemSize();  // in bytes
    const int rows = left.rows, cols = left.cols;

    // allocate input buffers
    OclMem l_mem;
    OCL_GUARD_RET(l_mem = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                         memory_size, left.data, &ret));
    OclMem r_mem;
    OCL_GUARD_RET(r_mem = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                         memory_size, right.data, &ret));

    // allocate intermediate buffers
    // for box blur:
    OclMem l_mean_mem;
    OCL_GUARD_RET(l_mean_mem =
                      clCreateBuffer(e.p.context, CL_MEM_READ_WRITE, memory_size, nullptr, &ret));
    OclMem r_mean_mem;
    OCL_GUARD_RET(r_mean_mem =
                      clCreateBuffer(e.p.context, CL_MEM_READ_WRITE, memory_size, nullptr, &ret));

    // for disparity:
    OclMem map_r2l_mem;
    OCL_GUARD_RET(map_r2l_mem =
                      clCreateBuffer(e.p.context, CL_MEM_READ_WRITE, memory_size, nullptr, &ret));

    // allocate output buffers
    cv::Mat map_l2r = cv::Mat::zeros(left.size(), CV_8UC1);
    REQUIRE(memory_size == (map_l2r.total() * map_l2r.elemSize()));
    OclMem map_l2r_mem;
    OCL_GUARD_RET(map_l2r_mem = clCreateBuffer(e.p.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                               memory_size, map_l2r.data, &ret));

    // execute kernels

    // use global sizes
    size_t work_group_sizes[] = {1, 256};
    // work items number must be divisible (no remainder) by the size of the work group
    size_t work_items_sizes[] = {
        size_t(std::ceil(double(rows) / work_group_sizes[0])) * work_group_sizes[0],
        size_t(std::ceil(double(cols) / work_group_sizes[1])) * work_group_sizes[1],
    };

    // box blur:

    // params: __global uchar* out, __global const uchar* in, int rows, int cols, int k_size

    // run for left image
    {
        ocl_set_kernel_args(e.kernels[0], l_mean_mem.memory, l_mem.memory, rows, cols,
                            stereo_common::MAX_WINDOW);

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 0);
    }

    // run for right image
    {
        ocl_set_kernel_args(e.kernels[0], r_mean_mem.memory, r_mem.memory);
        // other params remain as is

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 0);
    }

    // make disparity maps:

    // params: __global uchar* out, __global const uchar* left, __global const uchar* left_mean,
    //         __global const uchar* right, __global const uchar* right_mean, int rows, int cols,
    //         int window_size, int d_first, int d_last

    // run for l2r
    {
        ocl_set_kernel_args(e.kernels[1], map_l2r_mem.memory, l_mem.memory, l_mean_mem.memory,
                            r_mem.memory, r_mean_mem.memory, rows, cols, stereo_common::MAX_WINDOW,
                            -disparity, 0);

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 1);
    }

    // run for r2l
    {
        ocl_set_kernel_args(e.kernels[1], map_r2l_mem.memory, r_mem.memory, r_mean_mem.memory,
                            l_mem.memory, l_mean_mem.memory);
        ocl_set_kernel_args<8, 9>(e.kernels[1], 0, disparity);
        // other params remain as is

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 1);
    }

    // cross check maps:
    {
        // params: __global uchar* l2r, __global const uchar* r2l, int rows, int cols, int disparity
        ocl_set_kernel_args(e.kernels[2], map_l2r_mem.memory, map_r2l_mem.memory, rows, cols,
                            disparity);

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 2);
    }

    // fill occlusions:
    {
        // params: __global uchar* data, int rows, int cols, int disparity
        ocl_set_kernel_args(e.kernels[3], map_l2r_mem.memory, rows, cols, disparity);

        // special 1d sizes for fill occlusions
        size_t work_group_sizes[] = {2};
        // work items number must be divisible (no remainder) by the size of the work group
        size_t work_items_sizes[] = {
            size_t(std::ceil(double(rows) / work_group_sizes[0])) * work_group_sizes[0],
        };

        e.run_nd_range(1, work_items_sizes, work_group_sizes, 3);
    }

    // read output back into this process' memory
    OCL_GUARD_RET(clEnqueueMapBuffer(e.queue, map_l2r_mem, CL_TRUE, CL_MAP_READ, 0, memory_size, 0,
                                     nullptr, nullptr, &ret));

    return map_l2r;
}
}  // namespace stereo_ocl_base
