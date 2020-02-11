#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <utility>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// OpenCL image processing:
#include "rgb2gray_blur_ocl_inl.hpp"
// OpenCL optimized image processing:
#include "rgb2gray_blur_ocl_opt_inl.hpp"

#include "common/measure.hpp"
#include "common/utils.hpp"

namespace {
void print_usage(const char* program_name) {
    PRINTLN("Usage: " + std::string(program_name) + " [CL_PLATFORM_ID CL_DEVICE_ID]");
}
template<typename F>
std::pair<std::string, std::string> check_run(F f, int platform_id, int device_id, cv::Size size) {
    std::uint64_t res = 0;  // in microseconds
    try {
        res = f(platform_id, device_id, size);
    } catch (const std::exception& e) {
        PRINTLN(e.what());
        return {"FAIL", "0"};
    } catch (...) {
        PRINTLN("UNKNOWN ERROR HAPPENED");
        return {"FAIL", "0"};
    }
    return {"OK", std::to_string(double(res) / 1000) + " msec"};
}

constexpr const std::size_t ITERS = 1000;

std::map<std::string, std::uint64_t (*)(int, int, cv::Size)> ALL_PERF_TESTS = {};
#define PERF(suffix)                                                                               \
    ALL_PERF_TESTS["TEST_" #suffix] = [](int platform_id, int device_id,                           \
                                         cv::Size size) /* test body starts here */

void declare_tests() {
    PERF(OCL_BASELINE) {
        cv::Mat rgb(size, CV_8UC3);
        cv::randu(rgb, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        std::uint64_t time =
            measure(ITERS, [&]() { process_rgb_ocl(rgb, platform_id, device_id); });
        return time / ITERS;
    };

    PERF(OCL_OPTIMIZED) {
        cv::Mat rgb(size, CV_8UC3);
        cv::randu(rgb, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
        std::uint64_t time =
            measure(ITERS, [&]() { process_rgb_ocl_opt(rgb, platform_id, device_id); });
        return time / ITERS;
    };
}

}  // namespace

namespace std {
string to_string(cv::Size size) {
    std::stringstream ss;
    ss << size;
    return ss.str();
}
}  // namespace std

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

    declare_tests();

    cv::Size sizes[] = {cv::Size(4096, 2160), cv::Size(1920, 1080), cv::Size(640, 480)};
    for (const auto& it : ALL_PERF_TESTS) {
        const auto& name = it.first;
        const auto& perf_test = it.second;

        for (const auto& size : sizes) {
            auto res = check_run(perf_test, platform_id, device_id, size);
            PRINTLN("RUN [" + name + std::to_string(size) + "]");
            PRINTLN("TIME: " + res.second);
            PRINTLN("RUN [" + name + "]: " + res.first);
        }
    }

    return 0;
}
