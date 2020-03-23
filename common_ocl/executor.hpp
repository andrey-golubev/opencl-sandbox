#pragma once

#include "common/require.hpp"
#include "common_ocl/utils.hpp"

#include <string>
#include <utility>
#include <vector>

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
    std::vector<cl_kernel> kernels = {};

    OclExecutor(OclPrimitives&& prims, const std::string& program_str,
                std::vector<const char*>&& kernel_names)
        : p(std::move(prims)) {
        // create queue
        // TODO: clCreateCommandQueue is deprecated since (?) OpenCL 1.2 - but useful for CUDA
        OCL_GUARD_RET(queue = clCreateCommandQueue(p.context, p.device_id, 0, &ret));

        // create program
        const char* programs[] = {program_str.c_str()};
        const size_t sizes[] = {program_str.size()};
        OCL_GUARD_RET(program = clCreateProgramWithSource(p.context, 1, programs, sizes, &ret));
        OCL_GUARD_CUSTOM(
            clBuildProgram(program, 1, &p.device_id, nullptr, nullptr, nullptr),
            [&](const char* expr) {
                std::stringstream ss;
                ss << "Command '" << expr << "' failed\n";
                size_t str_size = 0;
                OCL_GUARD(clGetProgramBuildInfo(program, p.device_id, CL_PROGRAM_BUILD_LOG, 0,
                                                nullptr, &str_size));
                std::string str(str_size, '\0');
                OCL_GUARD(clGetProgramBuildInfo(program, p.device_id, CL_PROGRAM_BUILD_LOG,
                                                str.size(), &str[0], nullptr));
                ss << str << std::endl;
                return ss.str();
            });

        // create kernel
        for (const auto& name : kernel_names) {
            OCL_GUARD_RET(kernels.emplace_back(clCreateKernel(program, name, &ret)));
        }
    }

    OclExecutor(const OclExecutor& other) = delete;
    OclExecutor(OclExecutor&& other) {
        OCL_MOVE_STRUCT(p, other.p);
        OCL_MOVE_PTR(queue, other.queue);
        OCL_MOVE_PTR(program, other.program);
        kernels = std::move(other.kernels);
    }

    OclExecutor& operator=(const OclExecutor& other) = delete;
    OclExecutor& operator=(OclExecutor&& other) {
        OCL_MOVE_STRUCT(p, other.p);
        OCL_MOVE_PTR(queue, other.queue);
        OCL_MOVE_PTR(program, other.program);
        kernels = std::move(other.kernels);
        return *this;
    }

    void run_nd_range(cl_uint work_dim, const size_t* work_items_sizes,
                      const size_t* work_group_sizes, size_t kernel_idx) {
        REQUIRE(kernel_idx < kernels.size());
        OCL_GUARD(clEnqueueNDRangeKernel(queue, kernels[kernel_idx], work_dim, nullptr,
                                         work_items_sizes, work_group_sizes, 0, nullptr, nullptr));
    }

    ~OclExecutor() {
        if (queue != nullptr) {
            OCL_GUARD(clFlush(queue));
            OCL_GUARD(clFinish(queue));
        }
        for (auto& kernel : kernels) {
            if (kernel == nullptr)
                continue;
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

template<int I, int... Is, typename Arg, typename... Args>
static void ocl_set_kernel_args(cl_kernel k, Arg head, Args... tail) {
    ocl_set_kernel_args<I>(k, head);
    ocl_set_kernel_args<Is...>(k, tail...);
}

template<int I, typename Arg> static void ocl_set_kernel_args(cl_kernel k, Arg arg) {
    OCL_GUARD(clSetKernelArg(k, I, sizeof(Arg), &arg));
}

template<typename... Args, int... Is>
static void ocl_set_kernel_args_medium(cl_kernel k, std::integer_sequence<int, Is...> seq,
                                       Args... args) {
    ocl_set_kernel_args<Is...>(k, args...);
}

template<typename... Args> static void ocl_set_kernel_args(cl_kernel k, Args... args) {
    ocl_set_kernel_args_medium(k, std::make_integer_sequence<int, sizeof...(Args)>{}, args...);
}
