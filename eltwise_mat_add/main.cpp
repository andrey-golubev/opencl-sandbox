#include <cstdint>
#include <iostream>
#include <map>
#include <random>

#include <opencv2/core.hpp>

// CPP version of matrix summation:
#include "mat_add_cpp_inl.hpp"
// OpenCL version of matrix summation:
#include "mat_add_ocl_inl.hpp"

#include "common/utils.hpp"

int main(int argc, char* argv[]) {
    constexpr const int rows = 100;
    constexpr const int cols = 100;

    int platform_id = 0, device_id = 0;

    if (argc != 1) {

        if (argc != 3) {
            PRINTLN("Usage: " + std::string(argv[0]) + " [PLATFORM_ID] [DEVICE_ID]");
            PRINTLN("If platform and/or device ids are specified. The mode is quiet - no OpenCL "
                    "info reported");
            return 1;
        }

        platform_id = std::stoi(argv[1]);
        device_id = std::stoi(argv[2]);
    }

    PRINTLN("-----");
    PRINTLN("For OpenCL: using platform #" + std::to_string(platform_id) + " and device #" +
            std::to_string(device_id));
    PRINTLN("-----\n");

    cv::Mat a(cv::Size(rows, cols), CV_8UC3);
    cv::randu(a, cv::Scalar(0, 0, 0), cv::Scalar(125, 125, 125));

    cv::Mat b(cv::Size(rows, cols), CV_8UC3);
    cv::randu(b, cv::Scalar(0, 0, 0), cv::Scalar(125, 125, 125));

    // print OpenCL device information, etc.
    print_cl_info();

    // run CPP sum
    auto cpp_res = eltwise_add_ocv(a, b);
    // run OCL sum
    auto ocl_res = eltwise_add_ocl(a, b, platform_id, device_id);

    // compare results
    REQUIRE(cv::countNonZero(cpp_res != ocl_res) == 0);

    // compare performance:
    std::map<std::string, std::int64_t> perf_results = {};
    int iters = 10000;
    perf_results["ocv"] = measure(iters, [&]() { eltwise_add_ocv(a, b); });
    perf_results["ocl"] = measure(iters, [&]() { eltwise_add_ocl(a, b, platform_id, device_id); });
    perf_results["cpp"] = measure(iters, [&]() { eltwise_add_cpp(a, b); });

    // report performance numbers:
    for (const auto& e : perf_results) {
        PRINTLN("Average time (in microsec) per " + std::to_string(iters) + " iters for " +
                e.first + ": " + std::to_string(double(e.second) / iters));
    }

    return 0;
}
