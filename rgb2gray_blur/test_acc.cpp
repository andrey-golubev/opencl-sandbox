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

std::map<std::string, void (*)(int, int)> tests_map = {};
#define TEST(suffix)                                                                               \
    tests_map["test_" #suffix] = [](int platform_id, int device_id) /* test body starts here */

void declare_tests() {
    TEST(123) {
        cv::Mat rgb(cv::Size(12, 12), CV_8UC3);
        cv::randu(rgb, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

        cv::Mat ocv = process_rgb_cpp(rgb);
        cv::Mat ocl = process_rgb_ocl(rgb, platform_id, device_id);

        // TODO: not always consistent - implementations are too different?
        REQUIRE(equal_with_tolerance(ocv, ocl, 2));
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
        platform_id = std::stoi(argv[2]);
        device_id = std::stoi(argv[3]);
    }

    PRINTLN("-----");
    PRINTLN("For OpenCL: using platform #" + std::to_string(platform_id) + " and device #" +
            std::to_string(device_id));
    PRINTLN("-----\n");

    declare_tests();

    for (const auto& it : tests_map) {
        const auto& name = it.first;
        const auto& test = it.second;
        PRINTLN("RUN [" + name + "]");
        auto res = check_run(test, platform_id, device_id);
        PRINTLN("RUN [" + name + "]: " + res);
    }

    return 0;
}
