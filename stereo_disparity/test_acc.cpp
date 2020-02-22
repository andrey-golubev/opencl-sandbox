#include <cstdint>
#include <iostream>
#include <map>
#include <random>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>

// CPP stereo algo:
#include "stereo_disparity_cpp_inl.hpp"

// PFM image reader:
#include "pfm_reader.hpp"

#include "common/threading.hpp"
#include "common/utils.hpp"

namespace {
void print_usage(const char* program_name) {
    PRINTLN("Usage: " + std::string(program_name) +
            " TEST_DATA_FOLDER [CL_PLATFORM_ID CL_DEVICE_ID]");
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
    cv::Size(1920, 1080), cv::Size(640, 480), cv::Size(189, 279), cv::Size(12, 12), cv::Size(5, 5),
};

cv::Mat test_make_mean(const cv::Mat& in, int k_size) {
    cv::Mat out;
    cv::boxFilter(in, out, -1, cv::Size(k_size, k_size));
    return out;
}

std::string TEST_DATA_FOLDER = "";

void declare_tests() {
    TEST(BOX_BLUR_CPP) {
        for (const auto& size : TEST_SIZES) {
            cv::Mat in(size, CV_8UC1);
            cv::randu(in, cv::Scalar(0), cv::Scalar(255));

            cv::Mat ocv;
            cv::Mat cpp = cv::Mat::zeros(in.size(), in.type());

            for (int k_size : {5, 9}) {
                cv::boxFilter(in, ocv, -1, cv::Size(k_size, k_size));
                stereo_cpp_base::box_blur(in.data, cpp.data, in.rows, in.cols, k_size);
                REQUIRE(cv::countNonZero(ocv != cpp) == 0);
            }
        }
    };
    TEST(DISPARITY_MAP_SANITY) {
        for (auto append : {"/backpack/", "/umbrella/"}) {
            std::string folder = TEST_DATA_FOLDER + append;
            cv::Mat left_img = cv::imread(folder + "im0.png");
            cv::Mat right_img = cv::imread(folder + "im1.png");

            cv::Size resize_to(100, 100);  // resized down for increased speed
            {
                cv::resize(left_img, left_img, resize_to);
                cv::resize(right_img, right_img, resize_to);
            }

            cv::Mat left, right;
            cv::cvtColor(left_img, left, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right_img, right, cv::COLOR_BGR2GRAY);

            // test code:
            cv::Mat cpp_disp;
            int max_disp = 50;
            cpp_disp = stereo_cpp_base::stereo_compute_disparity(left, right, max_disp);

            cv::Mat zero = cv::Mat::zeros(cpp_disp.size(), cpp_disp.type());

            REQUIRE(cpp_disp.type() == CV_8UC1);
            REQUIRE(cv::countNonZero(zero == cpp_disp) < (cpp_disp.rows * cpp_disp.cols));
        }
    };
}

bool test_disabled(const std::string& test_name) {
    return std::string::npos != test_name.find("DISABLED");
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    int platform_id = 0, device_id = 0;
    if (argc == 4) {
        platform_id = std::stoi(argv[2]);
        device_id = std::stoi(argv[3]);
    }

    PRINTLN("-----");
    PRINTLN("For OpenCL: using platform #" + std::to_string(platform_id) + " and device #" +
            std::to_string(device_id));
    PRINTLN("-----\n");

    TEST_DATA_FOLDER = std::string(argv[1]) + "/stereo/";

    declare_tests();

    for (const auto& it : ALL_TESTS) {
        const auto& name = it.first;
        if (test_disabled(name)) {
            continue;
        }

        const auto& test = it.second;
        PRINTLN("RUN [" + name + "]");
        auto res = check_run(test, platform_id, device_id);
        PRINTLN("RUN [" + name + "]: " + res);
    }

    return 0;
}
