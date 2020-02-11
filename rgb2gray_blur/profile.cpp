#include <cstdint>
#include <iostream>
#include <map>
#include <random>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// OpenCL image processing:
#define NO_MAP_BUFFER
#include "rgb2gray_blur_ocl_inl.hpp"
#include "rgb2gray_blur_ocl_opt_inl.hpp"

#include "common/measure.hpp"
#include "common/utils.hpp"

namespace {
void print_usage(const char* program_name) {
    PRINTLN("Usage: " + std::string(program_name) + " [CL_PLATFORM_ID CL_DEVICE_ID]");
}

constexpr const size_t ITERS = 10000;
}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 1 && argc != 3) {
        print_usage(argv[0]);
        return 1;
    }

    int platform_id = 0, device_id = 0;
    if (argc == 3) {
        platform_id = std::stoi(argv[1]);
        device_id = std::stoi(argv[2]);
    }

    PRINTLN("-----");
    PRINTLN("For OpenCL: using platform #" + std::to_string(platform_id) + " and device #" +
            std::to_string(device_id));
    PRINTLN("-----\n");

    // Read input image:
    cv::Mat rgb(cv::Size(1920, 1080), CV_8UC3);
    cv::randu(rgb, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    // Run OpenCL baseline:
    // auto baseline_time = measure(ITERS, [&]() { process_rgb_ocl(rgb, platform_id, device_id); });

    auto optimized_time =
        measure(ITERS, [&]() { process_rgb_ocl_opt(rgb, platform_id, device_id); });

    // PRINTLN("BASELINE: " + std::to_string(baseline_time / ITERS));
    PRINTLN("OPTIMIZED: " + std::to_string(optimized_time / ITERS));

    return 0;
}
