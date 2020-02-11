#include <cstdint>
#include <iostream>
#include <map>
#include <random>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// CPP image processing:
#include "rgb2gray_blur_cpp_inl.hpp"
// OpenCL image processing:
#include "rgb2gray_blur_ocl_inl.hpp"
// OpenCL optimized image processing:
#include "rgb2gray_blur_ocl_opt_inl.hpp"

#include "common/utils.hpp"
#include "equal.hpp"

namespace {
void print_usage(const char* program_name) {
    PRINTLN("Usage: " + std::string(program_name) + " [CL_PLATFORM_ID CL_DEVICE_ID]");
}
template<typename F> std::string check_run(F f, int platform_id, int device_id) {
    try {
        f(platform_id, device_id);
    } catch (const std::exception& e) {
        PRINTLN(e.what());
        return "FAIL";
    } catch (...) {
        PRINTLN("UNKNOWN ERROR HAPPENED");
        return "FAIL";
    }
    return "OK";
}

std::map<std::string, void (*)(int, int)> ALL_TESTS = {};
#define TEST(suffix)                                                                               \
    ALL_TESTS["TEST_" #suffix] = [](int platform_id, int device_id) /* test body starts here */

const cv::Size TEST_SIZES[] = {
    cv::Size(4096, 2160), cv::Size(1920, 1080), cv::Size(640, 480),
    cv::Size(189, 279),   cv::Size(12, 12),     cv::Size(5, 5),
};

void declare_tests() {
    TEST(FULL_PIPELINE) {
        for (const auto& size : TEST_SIZES) {
            cv::Mat rgb(size, CV_8UC3);
            cv::randu(rgb, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

            cv::Mat cpp = process_rgb_cpp(rgb);
            cv::Mat ocl = process_rgb_ocl(rgb, platform_id, device_id);

            REQUIRE(equal_with_tolerance(cpp, ocl, 2));
        }
    };

    TEST(FULL_PIPELINE_OPT_OCL) {
        for (const auto& size : TEST_SIZES) {
            cv::Mat rgb(size, CV_8UC3);
            cv::randu(rgb, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

            cv::Mat cpp = process_rgb_cpp(rgb);
            cv::Mat ocl = process_rgb_ocl_opt(rgb, platform_id, device_id);

            REQUIRE(equal_with_tolerance(cpp, ocl, 2));
        }
    };
}

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

    declare_tests();

    for (const auto& it : ALL_TESTS) {
        const auto& name = it.first;
        const auto& test = it.second;
        PRINTLN("RUN [" + name + "]");
        auto res = check_run(test, platform_id, device_id);
        PRINTLN("RUN [" + name + "]: " + res);
    }

    return 0;
}
