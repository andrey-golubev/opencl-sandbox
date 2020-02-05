#pragma once

#include "common_ocl/executor.hpp"
#include "common_ocl/utils.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <iostream>
#include <string>

#include <CL/cl.h>

namespace {

const std::string program_str(R"(
__kernel void add_uchar(__global const uchar* a,
                        __global const uchar* b,
                        __global uchar* out,
                        int rows,
                        int cols,
                        int chan) {
    int i = get_global_id(0);  // row index
    int j = get_global_id(1);  // col index

    // process one pixel (but all channels)
    for (int c = 0; c < chan; ++c) {
        int idx = i * cols * chan + j * chan + c;
        out[idx] = a[idx] + b[idx];
    }
}
)");

}  // namespace

// interface:
cv::Mat eltwise_add_ocl(const cv::Mat& a, const cv::Mat& b, size_t platform_idx = 0,
                        size_t device_idx = 0) {
    static OclExecutor e(OclPrimitives(platform_idx, device_idx), program_str, {"add_uchar"});

    // allocate input buffers
    cl_mem aobj = nullptr;
    OCL_GUARD_RET(aobj = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        a.total() * a.elemSize(), a.data, &ret));

    cl_mem bobj = nullptr;
    OCL_GUARD_RET(bobj = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        b.total() * b.elemSize(), b.data, &ret));

    // allocate output buffers
    cv::Mat out = cv::Mat::zeros(a.size(), a.type());
    cl_mem outobj = nullptr;
    OCL_GUARD_RET(outobj = clCreateBuffer(e.p.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                          out.total() * out.elemSize(), out.data, &ret));

    // set kernel parameters (in/out)
    OCL_GUARD(clSetKernelArg(e.kernels[0], 0, sizeof(cl_mem), &aobj));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 1, sizeof(cl_mem), &bobj));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 2, sizeof(cl_mem), &outobj));
    int chan = a.channels();
    OCL_GUARD(clSetKernelArg(e.kernels[0], 3, sizeof(int), &a.rows));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 4, sizeof(int), &a.cols));
    OCL_GUARD(clSetKernelArg(e.kernels[0], 5, sizeof(int), &chan));

    // execute kernel
    size_t work_group_sizes[] = {10, 10};
    // work items number must be divisible (no remainder) by the size of the work group
    size_t work_items_sizes[] = {
        size_t(std::ceil(double(a.rows) / work_group_sizes[0])) * work_group_sizes[0],
        size_t(std::ceil(double(a.cols) / work_group_sizes[1])) * work_group_sizes[1],
    };

    e.run_nd_range(2, work_items_sizes, work_group_sizes, 0);

    // read output back into this process' memory
    OCL_GUARD_RET(clEnqueueMapBuffer(e.queue, outobj, CL_TRUE, CL_MAP_READ, 0,
                                     out.total() * out.elemSize(), 0, nullptr, nullptr, &ret));

    // release memory objects
    OCL_GUARD(clReleaseMemObject(outobj));
    OCL_GUARD(clReleaseMemObject(bobj));
    OCL_GUARD(clReleaseMemObject(aobj));

    return out;
}
