#include <cstdint>
#include <iostream>
#include <map>
#include <random>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>

// CPP zncc:
#include "zncc_cpp_inl.hpp"

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
    cv::Size(4096, 2160), cv::Size(1920, 1080), cv::Size(640, 480),
    cv::Size(189, 279),   cv::Size(12, 12),     cv::Size(5, 5),
};

std::string TEST_DATA_FOLDER = "";

void declare_tests() {
    TEST(DISABLED_BOX_BLUR_CPP) {
        for (const auto& size : TEST_SIZES) {
            cv::Mat in(size, CV_8UC1);
            cv::randu(in, cv::Scalar(0), cv::Scalar(255));

            cv::Mat ocv;
            cv::Mat cpp = cv::Mat::zeros(in.size(), in.type());

            for (int k_size : {5, 9}) {
                cv::boxFilter(in, ocv, -1, cv::Size(k_size, k_size));
                box_blur(in.data, cpp.data, in.rows, in.cols, k_size);
                REQUIRE(cv::countNonZero(ocv != cpp) == 0);
            }
        }
    };
    TEST(DISPARITY_BACKPACK) {
        std::string backpack_folder = TEST_DATA_FOLDER + "/stereo/backpack/";
        cv::Mat left_img = cv::imread(backpack_folder + "im0.png");
        cv::Mat right_img = cv::imread(backpack_folder + "im1.png");
        cv::Mat disp_l2r = cv::imread(backpack_folder + "disp0.pfm", cv::IMREAD_ANYDEPTH);
        cv::Mat disp_r2l = cv::imread(backpack_folder + "disp1.pfm", cv::IMREAD_ANYDEPTH);

        cv::resize(left_img, left_img, cv::Size(640, 480));
        cv::resize(right_img, right_img, cv::Size(640, 480));

        cv::resize(disp_l2r, disp_l2r, cv::Size(640, 480));
        cv::resize(disp_r2l, disp_r2l, cv::Size(640, 480));

        cv::Mat cpp_l2r, cpp_r2l;
        std::tie(cpp_l2r, cpp_r2l) =
            stereo_compute_disparities_impl(left_img, right_img, {-10, 10});

        cv::imshow("Computed disparity L2R", cpp_l2r);
        cv::imshow("Computed disparity R2L", cpp_r2l);
        cv::imshow("GT disparity L2R", disp_l2r);
        cv::imshow("GT disparity R2L", disp_r2l);
        cv::waitKey();

        // REQUIRE(cv::countNonZero(disp_l2r != cpp_l2r) == 0);
        // REQUIRE(cv::countNonZero(disp_r2l != cpp_r2l) == 0);
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

    TEST_DATA_FOLDER = argv[1];

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
