#include <cstdint>
#include <iostream>
#include <random>

#include <opencv2/core.hpp>

// CPP version of matrix summation:
#include "mat_add_cpp_inl.hpp"
// OpenCL version of matrix summation:
#include "mat_add_ocl_inl.hpp"

int main(int argc, char* argv[]) {
    constexpr const int rows = 100;
    constexpr const int cols = 100;

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

    REQUIRE(cv::countNonZero(cpp_res != ocl_res) == 0);
    return 0;
}
