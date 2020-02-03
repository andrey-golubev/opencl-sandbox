#pragma once

#include "common/require.hpp"
#include "common_ocl/utils.hpp"

struct OclPrimitives {
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_context context = nullptr;

    OclPrimitives(size_t platform_idx = 0, size_t device_idx = 0) {
        // choose platform
        auto pnum = get_cl_num_platforms();
        REQUIRE(platform_idx < pnum);
        std::vector<cl_platform_id> platforms(pnum, nullptr);
        OCL_GUARD(clGetPlatformIDs(pnum, platforms.data(), nullptr));
        platform_id = platforms[platform_idx];

        // choose platform
        auto dnum = get_cl_num_devices(platform_id, CL_DEVICE_TYPE_ALL);
        REQUIRE(device_idx < dnum);
        std::vector<cl_device_id> devices(dnum, nullptr);
        OCL_GUARD(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, dnum, devices.data(), nullptr));
        device_id = devices[device_idx];

        // create context
        OCL_GUARD_RET(context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret));
    }

    OclPrimitives(const OclPrimitives& other) = delete;
    OclPrimitives(OclPrimitives&& other) {
        OCL_MOVE_PTR(platform_id, other.platform_id);
        OCL_MOVE_PTR(device_id, other.device_id);
        OCL_MOVE_PTR(context, other.context);
    }

    OclPrimitives& operator=(const OclPrimitives& other) = delete;
    OclPrimitives& operator=(OclPrimitives&& other) {
        OCL_MOVE_PTR(platform_id, other.platform_id);
        OCL_MOVE_PTR(device_id, other.device_id);
        OCL_MOVE_PTR(context, other.context);
        return *this;
    }

    ~OclPrimitives() {
        if (context != nullptr) {
            OCL_GUARD(clReleaseContext(context));
        }
    }
};

struct OclExecutor {
    OclPrimitives p = {};

    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    OclExecutor(OclPrimitives&& prims, const char** programs, const size_t* sizes,
                const char* kernel_name)
        : p(std::move(prims)) {
        // create queue
        // TODO: clCreateCommandQueue is deprecated since (?) OpenCL 1.2 - but useful for CUDA
        OCL_GUARD_RET(queue = clCreateCommandQueue(p.context, p.device_id, 0, &ret));

        // create program
        OCL_GUARD_RET(program = clCreateProgramWithSource(p.context, 1, programs, sizes, &ret));
        OCL_GUARD(clBuildProgram(program, 1, &p.device_id, nullptr, nullptr, nullptr));

        // create kernel
        OCL_GUARD_RET(kernel = clCreateKernel(program, kernel_name, &ret));
    }

    OclExecutor(const OclExecutor& other) = delete;
    OclExecutor(OclExecutor&& other) {
        OCL_MOVE_STRUCT(p, other.p);
        OCL_MOVE_PTR(queue, other.queue);
        OCL_MOVE_PTR(program, other.program);
        OCL_MOVE_PTR(kernel, other.kernel);
    }

    OclExecutor& operator=(const OclExecutor& other) = delete;
    OclExecutor& operator=(OclExecutor&& other) {
        OCL_MOVE_STRUCT(p, other.p);
        OCL_MOVE_PTR(queue, other.queue);
        OCL_MOVE_PTR(program, other.program);
        OCL_MOVE_PTR(kernel, other.kernel);
        return *this;
    }

    ~OclExecutor() {
        if (queue != nullptr) {
            OCL_GUARD(clFlush(queue));
            OCL_GUARD(clFinish(queue));
        }
        if (kernel != nullptr) {
            OCL_GUARD(clReleaseKernel(kernel));
        }
        if (program != nullptr) {
            OCL_GUARD(clReleaseProgram(program));
        }
        if (queue != nullptr) {
            OCL_GUARD(clReleaseCommandQueue(queue));
        }
    }
};