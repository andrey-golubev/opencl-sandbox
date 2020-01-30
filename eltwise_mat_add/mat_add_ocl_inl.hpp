#pragma once

#include "ocl_utils/ocl_utils.hpp"

#include "opencv2/core.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include <CL/cl.h>

namespace {

const std::string program_str(R"(
__kernel void add_uchar(__global const uchar* a, __global const uchar* b, __global uchar* out) {
    int i = get_global_id(0);
    out[i] = a[i] + b[i];
}

__kernel void add_float(__global const float* a, __global const float* b, __global float* out) {
    int i = get_global_id(0);
    out[i] = a[i] + b[i];
})");

}  // namespace

// interface:
cv::Mat eltwise_add_ocl(const cv::Mat& a, const cv::Mat& b, size_t platform_idx = 0,
                        size_t device_idx = 0) {
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;

    // choose platform
    {
        auto pnum = get_cl_num_platforms();
        REQUIRE(platform_idx < pnum);
        std::vector<cl_platform_id> platforms(pnum, nullptr);
        OCL_GUARD(clGetPlatformIDs(pnum, platforms.data(), nullptr));
        platform_id = platforms[platform_idx];
    }

    // choose device
    {
        auto dnum = get_cl_num_devices(platform_id, CL_DEVICE_TYPE_ALL);
        REQUIRE(device_idx < dnum);
        std::vector<cl_device_id> devices(dnum, nullptr);
        OCL_GUARD(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, dnum, devices.data(), nullptr));
        device_id = devices[device_idx];
    }

    // create context
    cl_context context = nullptr;
    OCL_GUARD_RET(context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret));

    // create queue
    cl_command_queue queue = nullptr;
    // TODO: clCreateCommandQueue is deprecated since (?) OpenCL 1.2
    OCL_GUARD_RET(queue = clCreateCommandQueue(context, device_id, 0, &ret));

    // allocate input buffers
    cl_mem aobj = nullptr;
    OCL_GUARD_RET(aobj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        a.total() * a.elemSize(), a.data, &ret));

    cl_mem bobj = nullptr;
    OCL_GUARD_RET(bobj = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                        b.total() * b.elemSize(), b.data, &ret));

    // allocate output buffers
    cv::Mat out = cv::Mat::zeros(a.size(), a.type());
    cl_mem outobj = nullptr;
    OCL_GUARD_RET(outobj = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                          out.total() * out.elemSize(), out.data, &ret));

    // create program
    cl_program program = nullptr;
    {
        const char* programs[] = {program_str.c_str()};
        const size_t sizes[] = {program_str.size()};
        OCL_GUARD_RET(program = clCreateProgramWithSource(context, 1, programs, sizes, &ret));
        OCL_GUARD(clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr));
    }

    // select kernel
    cl_kernel kernel = nullptr;
    OCL_GUARD_RET(kernel = clCreateKernel(program, "add_uchar", &ret));

    // set kernel parameters (in/out)
    OCL_GUARD(clSetKernelArg(kernel, 0, sizeof(cl_mem), &aobj));
    OCL_GUARD(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bobj));
    OCL_GUARD(clSetKernelArg(kernel, 2, sizeof(cl_mem), &outobj));

    // execute kernel
    // OCL_GUARD(clEnqueueTask(queue, kernel, 0, nullptr, nullptr));
    size_t local_size = 64;
    size_t global_size = std::ceil(double(a.total() * a.channels()) / local_size) * local_size;
    OCL_GUARD(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0,
                                     nullptr, nullptr));

    // read output back into this process' memory
    OCL_GUARD_RET(clEnqueueMapBuffer(queue, outobj, CL_TRUE, CL_MAP_READ, 0,
                                     out.total() * out.elemSize(), 0, nullptr, nullptr, &ret));

    OCL_GUARD(clFlush(queue));
    OCL_GUARD(clFinish(queue));
    OCL_GUARD(clReleaseKernel(kernel));
    OCL_GUARD(clReleaseProgram(program));
    OCL_GUARD(clReleaseMemObject(aobj));
    OCL_GUARD(clReleaseMemObject(bobj));
    OCL_GUARD(clReleaseMemObject(outobj));
    OCL_GUARD(clReleaseCommandQueue(queue));
    OCL_GUARD(clReleaseContext(context));

    return a + b;
}
