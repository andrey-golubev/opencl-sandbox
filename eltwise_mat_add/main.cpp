#include <cstdint>
#include <iostream>
#include <random>

#include <opencv2/core.hpp>

// CPP version of matrix summation:
#include "mat_add_cpp_inl.hpp"
// OpenCL version of matrix summation:
#include "mat_add_ocl_inl.hpp"

int main(int argc, char* argv[]) {
    constexpr const int rows = 2;
    constexpr const int cols = 2;

    cv::Mat a(cv::Size(rows, cols), CV_8UC3);
    cv::randu(a, cv::Scalar(0, 0, 0), cv::Scalar(125, 125, 125));

    cv::Mat b(cv::Size(rows, cols), CV_8UC3);
    cv::randu(b, cv::Scalar(0, 0, 0), cv::Scalar(125, 125, 125));

    // print OpenCL device information, etc.
    print_cl_info();

    // run CPP sum
    auto cpp_res = eltwise_add_cpp(a, b);
    // run OCL sum
    auto ocl_res = eltwise_add_ocl(a, b);

    CV_Assert(cv::countNonZero(cpp_res != ocl_res) == 0);
    return 0;
}
