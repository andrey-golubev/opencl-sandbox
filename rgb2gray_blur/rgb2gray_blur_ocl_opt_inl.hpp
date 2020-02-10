#pragma once

#include "common_ocl/executor.hpp"

#include <opencv2/core.hpp>

namespace {
const std::string program_str_opt(R"(
__kernel void rgb2gray(__global const uchar* rgb, __global uchar* gray) {
    int i = get_global_id(0);  // pixel index
    const uchar r = rgb[3*i + 0]; const uchar g = rgb[3*i + 1]; const uchar b = rgb[3*i + 2];
    // formula: 0.3 * r + 0.59 * g + 0.11 * b
    gray[i] = 0.3 * r + 0.59 * g + 0.11 * b;
}

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

__kernel void moving_avg5x5(__global const uchar* gray, __global uchar* out, int rows, int cols) {
    const int idx_i = get_global_id(0);  // pixel index for row
    const int idx_j = get_global_id(1);  // pixel index for col

    // skip extra work items
    if (out_of_bounds(idx_i, 0, rows - 1) || out_of_bounds(idx_j, 0, cols - 1)) {
        return;
    }

    const int k_size = 5;
    const int center_shift = (k_size - 1) / 2;  // int(5 / 2)

    int sum = 0;
    for (int i = 0; i < k_size; ++i) {
        const int ii = fix(idx_i + i - center_shift, rows - 1);
        for (int j = 0; j < k_size; ++j) {
            const int jj = fix(idx_j + j - center_shift, cols - 1);
            sum += gray[ii * cols + jj];
        }
    }

    double dsum = sum;
    out[idx_i * cols + idx_j] = round(dsum / (k_size * k_size));
}

)");
}

cv::Mat rgb2gray_ocl_opt(cv::Mat rgb, size_t platform_idx = 0, size_t device_idx = 0) {
    static OclExecutor e(OclPrimitives(platform_idx, device_idx), program_str_opt, {"rgb2gray"});

    // allocate input buffers
    cl_mem rgbmem = nullptr;
    OCL_GUARD_RET(rgbmem = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                          rgb.total() * rgb.elemSize(), rgb.data, &ret));

    // allocate output buffers
    cv::Mat out = cv::Mat::zeros(rgb.size(), CV_8UC1);
    cl_mem outmem = nullptr;
    OCL_GUARD_RET(outmem = clCreateBuffer(e.p.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                          out.total() * out.elemSize(), out.data, &ret));

    // set kernel parameters (in/out)
    // rgb2gray:
    OCL_GUARD(clSetKernelArg(e.kernels[0], 0, sizeof(cl_mem), &rgbmem));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 1, sizeof(cl_mem), &outmem));

    // execute kernels:
    // rgb2gray:
    {
        size_t work_group_sizes[] = {10};
        // work items number must be divisible (no remainder) by the size of the work group
        size_t work_items_sizes[] = {
            size_t(std::ceil(double(rgb.total()) / work_group_sizes[0])) * work_group_sizes[0],
        };

        e.run_nd_range(1, work_items_sizes, work_group_sizes, 0);
    }

    // read output back into this process' memory
    OCL_GUARD_RET(clEnqueueMapBuffer(e.queue, outmem, CL_TRUE, CL_MAP_READ, 0,
                                     out.total() * out.elemSize(), 0, nullptr, nullptr, &ret));

    // release memory objects
    OCL_GUARD(clReleaseMemObject(outmem));
    OCL_GUARD(clReleaseMemObject(rgbmem));

    return out;
}

cv::Mat moving_avg_ocl_opt(cv::Mat gray, size_t platform_idx = 0, size_t device_idx = 0) {
    static OclExecutor e(OclPrimitives(platform_idx, device_idx), program_str_opt,
                         {"moving_avg5x5"});

    // allocate input buffers
    cl_mem graymem = nullptr;
    OCL_GUARD_RET(graymem = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                           gray.total() * gray.elemSize(), gray.data, &ret));

    // allocate output buffers
    cv::Mat out = cv::Mat::zeros(gray.size(), CV_8UC1);
    cl_mem outmem = nullptr;
    OCL_GUARD_RET(outmem = clCreateBuffer(e.p.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                          out.total() * out.elemSize(), out.data, &ret));

    // set kernel parameters (in/out)
    // moving_avg5x5:
    OCL_GUARD(clSetKernelArg(e.kernels[0], 0, sizeof(cl_mem), &graymem));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 1, sizeof(cl_mem), &outmem));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 2, sizeof(int), &gray.rows));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 3, sizeof(int), &gray.cols));

    // execute kernels:
    // moving_avg5x5:
    {
        size_t work_group_sizes[] = {10, 10};
        // work items number must be divisible (no remainder) by the size of the work group
        size_t work_items_sizes[] = {
            size_t(std::ceil(double(gray.rows) / work_group_sizes[0])) * work_group_sizes[0],
            size_t(std::ceil(double(gray.cols) / work_group_sizes[1])) * work_group_sizes[1],
        };

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 0);
    }

    // read output back into this process' memory
    OCL_GUARD_RET(clEnqueueMapBuffer(e.queue, outmem, CL_TRUE, CL_MAP_READ, 0,
                                     out.total() * out.elemSize(), 0, nullptr, nullptr, &ret));

    // release memory objects
    OCL_GUARD(clReleaseMemObject(outmem));
    OCL_GUARD(clReleaseMemObject(graymem));

    return out;
}

cv::Mat process_rgb_ocl_opt(cv::Mat rgb, size_t platform_idx = 0, size_t device_idx = 0) {
    static OclExecutor e(OclPrimitives(platform_idx, device_idx), program_str_opt,
                         {"rgb2gray", "moving_avg5x5"});

    // allocate input buffers
    cl_mem rgbmem = nullptr;
    OCL_GUARD_RET(rgbmem = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                          rgb.total() * rgb.elemSize(), rgb.data, &ret));

    // allocate intermediate buffers
    cl_mem graymem = nullptr;
    OCL_GUARD_RET(graymem = clCreateBuffer(e.p.context, CL_MEM_READ_WRITE,
                                           rgb.total() * rgb.elemSize1(), nullptr, &ret));

    // allocate output buffers
    cv::Mat out = cv::Mat::zeros(rgb.size(), CV_8UC1);
    cl_mem outmem = nullptr;
    OCL_GUARD_RET(outmem = clCreateBuffer(e.p.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                          out.total() * out.elemSize(), out.data, &ret));

    // set kernel parameters (in/out)
    // rgb2gray:
    OCL_GUARD(clSetKernelArg(e.kernels[0], 0, sizeof(cl_mem), &rgbmem));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 1, sizeof(cl_mem), &graymem));

    // moving_avg5x5:
    OCL_GUARD(clSetKernelArg(e.kernels[1], 0, sizeof(cl_mem), &graymem));
    OCL_GUARD(clSetKernelArg(e.kernels[1], 1, sizeof(cl_mem), &outmem));
    OCL_GUARD(clSetKernelArg(e.kernels[1], 2, sizeof(int), &rgb.rows));
    OCL_GUARD(clSetKernelArg(e.kernels[1], 3, sizeof(int), &rgb.cols));

    // execute kernels:

    // rgb2gray:
    {
        size_t work_group_sizes[] = {10};
        // work items number must be divisible (no remainder) by the size of the work group
        size_t work_items_sizes[] = {
            size_t(std::ceil(double(rgb.total()) / work_group_sizes[0])) * work_group_sizes[0],
        };

        e.run_nd_range(1, work_items_sizes, work_group_sizes, 0);
    }

    // moving_avg5x5:
    {
        size_t work_group_sizes[] = {3, 3};
        // work items number must be divisible (no remainder) by the size of the work group
        size_t work_items_sizes[] = {
            size_t(std::ceil(double(rgb.rows) / work_group_sizes[0])) * work_group_sizes[0],
            size_t(std::ceil(double(rgb.cols) / work_group_sizes[1])) * work_group_sizes[1],
        };

        e.run_nd_range(2, work_items_sizes, work_group_sizes, 1);
    }

    // read output back into this process' memory
    OCL_GUARD_RET(clEnqueueMapBuffer(e.queue, outmem, CL_TRUE, CL_MAP_READ, 0,
                                     out.total() * out.elemSize(), 0, nullptr, nullptr, &ret));

    // release memory objects
    OCL_GUARD(clReleaseMemObject(outmem));
    OCL_GUARD(clReleaseMemObject(graymem));
    OCL_GUARD(clReleaseMemObject(rgbmem));

    return out;
}
