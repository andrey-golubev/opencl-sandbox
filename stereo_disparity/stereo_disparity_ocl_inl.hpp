#pragma once

#include <cmath>
#include <utility>

#include <opencv2/core.hpp>

#include "common/utils.hpp"
#include "common_ocl/executor.hpp"
#include "common_ocl/utils.hpp"
#include "common_ocl/wrappers.hpp"

#include "stereo_common.hpp"

namespace stereo_ocl_base {

// TODO: defines must be taken from stereo_common instead of hard-coded in the string - replace
//       special symbols before providing string to OpenCL?
// TODO: optimize indices in box_blur and other kernels
const std::string program_str(R"(
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
#define EPS 0.0005
#define UNKNOWN_DISPARITY 0
#define MAX_WINDOW 11
#define MAX_BORDER 5

// kernels and subroutines:
__kernel void box_blur(__global uchar* out, __global const uchar* in, int rows, int cols,
                       int k_size) {
    const int idx_i = get_global_id(0);  // pixel index for row
    const int idx_j = get_global_id(1);  // pixel index for col

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1) || out_of_bounds(idx_j, 0, cols - 1)) {
        return;
    }

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

    out[idx_i * cols + idx_j] = round(multiplier * sum);
}

double zncc(__global const uchar* left, uchar l_mean, __global const uchar* right, uchar r_mean,
            int rows, int cols, int k_size, int idx_i, int idx_j, int d) {
    const int center_shift = (k_size - 1) / 2;

    int sum = 0;
    double std_left = 0.0, std_right = 0.0;

    for (int i = 0; i < k_size; ++i) {
        const int ii = fix(idx_i + i - center_shift, rows - 1);

        for (int j = 0; j < k_size; ++j) {
            const int jj1 = fix(idx_j + j - center_shift, cols - 1);
            const int jj2 = fix(idx_j + j - center_shift + d, cols - 1);

            const int left_pixel = (int)(left[ii * cols + jj1]) - (int)(l_mean);
            const int right_pixel = (int)(right[ii * cols + jj2]) - (int)(r_mean);

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

cv::Mat stereo_compute_disparity(const cv::Mat& left, const cv::Mat& right, int disparity,
                                 size_t platform_idx = 0, size_t device_idx = 0) {
    // sanity checks:
    REQUIRE(left.type() == CV_8UC1);
    REQUIRE(left.type() == right.type());
    REQUIRE(left.dims == 2);
    REQUIRE(left.rows == right.rows);
    REQUIRE(left.cols == right.cols);
    REQUIRE(disparity <= std::numeric_limits<uchar>::max());

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
    size_t work_group_sizes[] = {3, 3};
    // work items number must be divisible (no remainder) by the size of the work group
    size_t work_items_sizes[] = {
        size_t(std::ceil(double(rows) / work_group_sizes[0])) * work_group_sizes[0],
        size_t(std::ceil(double(cols) / work_group_sizes[1])) * work_group_sizes[1],
    };

    // box blur:

    // params: __global uchar* out, __global const uchar* in, int rows, int cols, int k_size

    // run for left image
    {
        OCL_GUARD(clSetKernelArg(e.kernels[0], 0, sizeof(cl_mem), &l_mean_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[0], 1, sizeof(cl_mem), &l_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[0], 2, sizeof(int), &rows));
        OCL_GUARD(clSetKernelArg(e.kernels[0], 3, sizeof(int), &cols));
        OCL_GUARD(clSetKernelArg(e.kernels[0], 4, sizeof(int), &stereo_common::MAX_WINDOW));

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 0);
    }

    // run for right image
    {
        OCL_GUARD(clSetKernelArg(e.kernels[0], 0, sizeof(cl_mem), &r_mean_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[0], 1, sizeof(cl_mem), &r_mem));
        // other params remain as is

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 0);
    }

    // make disparity maps:

    // params: __global uchar* out, __global const uchar* left, __global const uchar* left_mean,
    //         __global const uchar* right, __global const uchar* right_mean, int rows, int cols,
    //         int window_size, int d_first, int d_last

    // run for l2r
    {
        OCL_GUARD(clSetKernelArg(e.kernels[1], 0, sizeof(cl_mem), &map_l2r_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 1, sizeof(cl_mem), &l_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 2, sizeof(cl_mem), &l_mean_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 3, sizeof(cl_mem), &r_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 4, sizeof(cl_mem), &r_mean_mem));

        OCL_GUARD(clSetKernelArg(e.kernels[1], 5, sizeof(int), &rows));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 6, sizeof(int), &cols));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 7, sizeof(int), &stereo_common::MAX_WINDOW));

        const int d_first = -disparity, d_last = 0;
        OCL_GUARD(clSetKernelArg(e.kernels[1], 8, sizeof(int), &d_first));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 9, sizeof(int), &d_last));

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 1);
    }

    // run for r2l
    {
        OCL_GUARD(clSetKernelArg(e.kernels[1], 0, sizeof(cl_mem), &map_r2l_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 1, sizeof(cl_mem), &r_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 2, sizeof(cl_mem), &r_mean_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 3, sizeof(cl_mem), &l_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 4, sizeof(cl_mem), &l_mean_mem));

        const int d_first = 0, d_last = disparity;
        OCL_GUARD(clSetKernelArg(e.kernels[1], 8, sizeof(int), &d_first));
        OCL_GUARD(clSetKernelArg(e.kernels[1], 9, sizeof(int), &d_last));
        // other params remain as is

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 1);
    }

    // cross check maps:
    {
        // params: __global uchar* l2r, __global const uchar* r2l, int rows, int cols, int disparity
        OCL_GUARD(clSetKernelArg(e.kernels[2], 0, sizeof(cl_mem), &map_l2r_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[2], 1, sizeof(cl_mem), &map_r2l_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[2], 2, sizeof(int), &rows));
        OCL_GUARD(clSetKernelArg(e.kernels[2], 3, sizeof(int), &cols));
        OCL_GUARD(clSetKernelArg(e.kernels[2], 4, sizeof(int), &disparity));

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 2);
    }

    // fill occlusions:
    {
        // params: __global uchar* data, int rows, int cols, int disparity
        OCL_GUARD(clSetKernelArg(e.kernels[3], 0, sizeof(cl_mem), &map_l2r_mem));
        OCL_GUARD(clSetKernelArg(e.kernels[3], 1, sizeof(int), &rows));
        OCL_GUARD(clSetKernelArg(e.kernels[3], 2, sizeof(int), &cols));
        OCL_GUARD(clSetKernelArg(e.kernels[3], 3, sizeof(int), &disparity));

        // special 1d sizes for fill occlusions
        size_t work_group_sizes[] = {10};
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
