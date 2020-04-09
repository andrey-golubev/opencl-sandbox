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

#include "stereo_disparity_cpp_opt_inl.hpp"  // TODO: remove this include

namespace stereo_ocl_base {
inline std::string read_opencl_program() {
    static const std::string program_path = []() -> std::string {
        char delim =
#ifndef _WIN32
            '/';
#else
            '\\';
#endif
        const char* last_delimiter_pos = std::strrchr(__FILE__, delim);
        REQUIRE(last_delimiter_pos != nullptr);
        std::string root_dir(__FILE__, std::distance(__FILE__, last_delimiter_pos + 1));
        return root_dir + "stereo_disparity_kernels.cl";
    }();
    // TODO: the following is Linux specific:
    std::ifstream program_file(program_path);
    return std::string(std::istreambuf_iterator<char>(program_file),
                       std::istreambuf_iterator<char>());
}

template<typename T> void replace_param(std::string& string, std::string search_for, T param) {
    std::regex re(search_for);
    string = std::regex_replace(string, re, std::to_string(param));
}

inline void set_parameters(std::string& ocl_program) {
    replace_param(ocl_program, "<re:EPS>", stereo_common::EPS);
    replace_param(ocl_program, "<re:UNKNOWN_DISPARITY>", stereo_common::UNKNOWN_DISPARITY);
    replace_param(ocl_program, "<re:MAX_WINDOW>", stereo_common::MAX_WINDOW);
    replace_param(ocl_program, "<re:MAX_BORDER>", stereo_common::MAX_BORDER);
}

std::string get_program() {
    auto ocl_program = read_opencl_program();
    set_parameters(ocl_program);
    return ocl_program;
}

void _internal_stereo_compute_disparity(cv::Mat& out, const cv::Mat& left, const cv::Mat& right,
                                        int disparity, size_t platform_idx = 0,
                                        size_t device_idx = 0) {
    static OclExecutor e(
        OclPrimitives(platform_idx, device_idx), get_program(),
        {"box_blur", "make_disparity_map", "cross_check_disparity", "fill_occlusions_disparity"});

    constexpr const int border = stereo_common::MAX_BORDER;

    const size_t memory_size = left.total() * left.elemSize();  // in bytes
    const size_t memory_size_border = (left.rows + (border * 2)) * left.cols * left.elemSize();
    const int rows = left.rows, cols = left.cols;

    // allocate input buffers
    OclMem l_mem;
    OCL_GUARD_RET(l_mem = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                         memory_size_border, left.data, &ret));
    OclMem r_mem;
    OCL_GUARD_RET(r_mem = clCreateBuffer(e.p.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                         memory_size_border, right.data, &ret));

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
    OclMem map_l2r_mem;
    OCL_GUARD_RET(map_l2r_mem = clCreateBuffer(e.p.context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                               memory_size, out.data, &ret));

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
        size_t work_group_sizes[] = {256};
        // work items number must be divisible (no remainder) by the size of the work group
        size_t work_items_sizes[] = {
            size_t(std::ceil(double(rows) / work_group_sizes[0])) * work_group_sizes[0],
        };

        e.run_nd_range(1, work_items_sizes, work_group_sizes, 3);
    }

    // read output back into this process' memory
    OCL_GUARD_RET(clEnqueueMapBuffer(e.queue, map_l2r_mem, CL_TRUE, CL_MAP_READ, 0, memory_size, 0,
                                     nullptr, nullptr, &ret));
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
    REQUIRE(left.rows > stereo_common::MAX_BORDER);
    REQUIRE(left.cols > disparity);

    const int rows = left.rows, cols = left.cols;
    constexpr const int border = stereo_common::MAX_BORDER;

    // extend inputs with row borders
    cv::Mat in_left = stereo_cpp_opt::copy_make_border(left, border, 0);
    cv::Mat in_right = stereo_cpp_opt::copy_make_border(right, border, 0);

    // allocate output buffers
    cv::Mat map_l2r = cv::Mat::zeros(left.size(), CV_8UC1);

    constexpr const double arbitrary_scale = 0.05;
    // process k lines per OpenCL run - this appears to improve the speed drastically (why?)
    // TODO: figure out better approach
    const int k = std::max(1, int(std::round(arbitrary_scale * rows)));
    for (int y = 0; y < rows; y += k) {
        const int height = std::min(k, rows - y);

        // get ROI of inputs and output
        cv::Mat left_roi = in_left(cv::Rect(0, y, cols, height));
        cv::Mat right_roi = in_right(cv::Rect(0, y, cols, height));
        cv::Mat map_l2r_roi = map_l2r(cv::Rect(0, y, cols, height));

        // run OpenCL code for a slice
        _internal_stereo_compute_disparity(map_l2r_roi, left_roi, right_roi, disparity,
                                           platform_idx, device_idx);
    }

    return map_l2r;
}
}  // namespace stereo_ocl_base
