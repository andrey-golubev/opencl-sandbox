#pragma once

#include "common/utils.hpp"

#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <CL/cl.h>

void ocl_guard(int value, const char* expr, const char* file, int line) {
    std::stringstream ss;
    ss << "In file " << file << ", line " << line << ": " << expr << " == " << value
       << " and != CL_SUCCESS(0)";
    REQUIRE2(value == CL_SUCCESS, ss.str());
    return;
}
template<typename ErrorHandler>
void ocl_guard_custom(int value, const char* expr, ErrorHandler handle) {
    REQUIRE_CUSTOM(value == CL_SUCCESS, expr, handle);
    return;
}
#define OCL_GUARD(expr) ocl_guard((expr), #expr, __FILE__, __LINE__)
#define OCL_GUARD_RET(expr)                                                                        \
    {                                                                                              \
        cl_int ret = CL_SUCCESS;                                                                   \
        (expr); /* expecting `ret` to be updated here */                                           \
        OCL_GUARD(ret);                                                                            \
    }

#define OCL_GUARD_CUSTOM(expr, custom_call) ocl_guard_custom((expr), #expr, custom_call)

#define OCL_MOVE_PTR(to, from)                                                                     \
    { std::swap(to, from); }

#define OCL_MOVE_STRUCT(to, from)                                                                  \
    { std::swap(to, from); }

std::string type2str(cl_device_type t) {
    switch (t) {
    case CL_DEVICE_TYPE_CPU:
        return "CPU";
    case CL_DEVICE_TYPE_GPU:
        return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR:
        return "ACCELERATOR";
    }
    return "UNKNOWN";
}

cl_uint get_cl_num_platforms() {
    cl_uint ret_num_platforms = 0;
    OCL_GUARD(clGetPlatformIDs(0, nullptr, &ret_num_platforms));
    return ret_num_platforms;
}

cl_uint get_cl_num_devices(cl_platform_id platform_id, cl_device_type type) {
    cl_uint ret_num_devices = 0;
    OCL_GUARD(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &ret_num_devices));
    return ret_num_devices;
}

void print_info_impl(std::vector<std::pair<std::string, std::function<cl_int(std::string&)>>> in) {
    for (const auto& e : in) {
        const auto& prefix = e.first;
        const auto& f = e.second;
        // NB: str can be re-assigned inside, so wiser to create the string anew each time
        std::string str(256, '\0');
        OCL_GUARD(f(str));
        PRINTLN(prefix + str);
    }
}

void print_cl_info() {
    cl_uint num_platforms = get_cl_num_platforms();
    PRINTLN("Number of platforms: " + std::to_string(num_platforms));

    // 1. iterate over platforms:
    std::vector<cl_platform_id> platforms(num_platforms, nullptr);
    OCL_GUARD(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));
    for (auto platform_id : platforms) {
        REQUIRE(platform_id != nullptr);

        // 1.a. print platform information:
        PRINTLN("\nPlatform info:");
        std::string extra(2, ' ');
        PRINTLN(extra + "-----");

        print_info_impl({
            {extra + "Name: ",
             [&](std::string& str) {
                 return clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, str.size(), &str[0],
                                          nullptr);
             }},
            {extra + "Version: ",
             [&](std::string& str) {
                 return clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, str.size(), &str[0],
                                          nullptr);
             }},
            {extra + "Vendor: ",
             [&](std::string& str) {
                 return clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, str.size(), &str[0],
                                          nullptr);
             }},
            {extra + "Profile: ",
             [&](std::string& str) {
                 return clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, str.size(), &str[0],
                                          nullptr);
             }},
        });

        // 1.b. get devices
        cl_device_type type = CL_DEVICE_TYPE_ALL;
        cl_uint num_devices = get_cl_num_devices(platform_id, type);
        PRINTLN(extra + "Number of devices: " + std::to_string(num_devices));

        // 2. iterate over devices:
        PRINTLN("\n" + extra + "Device info:");

        std::vector<cl_device_id> devices(num_devices, nullptr);
        OCL_GUARD(clGetDeviceIDs(platform_id, type, num_devices, devices.data(), nullptr));
        for (auto device_id : devices) {
            std::string extra(4, ' ');
            PRINTLN(extra + "-----");
            PRINTLN(extra + "OpenCL device information:");

            print_info_impl({
                {extra + "Name: ",
                 [&](std::string& str) {
                     return clGetDeviceInfo(device_id, CL_DEVICE_NAME, str.size(), &str[0],
                                            nullptr);
                 }},
                {extra + "Version: ",
                 [&](std::string& str) {
                     return clGetDeviceInfo(device_id, CL_DEVICE_VERSION, str.size(), &str[0],
                                            nullptr);
                 }},
                {extra + "Type: ",
                 [&](std::string& str) {
                     auto ret =
                         clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
                     str = type2str(type);
                     return ret;
                 }},
                {extra + "Profile: ",
                 [&](std::string& str) {
                     return clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, str.size(), &str[0],
                                            nullptr);
                 }},
            });
        }
    }
}
