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
    cv::Size(4096, 2160), cv::Size(1920, 1080), cv::Size(640, 480),
    cv::Size(189, 279),   cv::Size(12, 12),     cv::Size(5, 5),
};

cv::Mat test_make_mean(const cv::Mat& in, int k_size) {
    cv::Mat out;
    cv::boxFilter(in, out, -1, cv::Size(k_size, k_size));
    return out;
}

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
    TEST(DISPARITY_MAP_L2R) {
        std::string backpack_folder = TEST_DATA_FOLDER + "/stereo/backpack/";
        cv::Mat left_img = cv::imread(backpack_folder + "im0.png");
        cv::Mat right_img = cv::imread(backpack_folder + "im1.png");

        std::unique_ptr<float[]> raw_disp;
        cv::Mat disp_l2r = cv::imread(backpack_folder + "disp0.pfm", cv::IMREAD_LOAD_GDAL);
        {
            PFMReader reader;
            raw_disp = std::move(reader.read<float>(backpack_folder + "disp0.pfm"));
            REQUIRE(raw_disp != nullptr);
            disp_l2r = cv::Mat(reader.rows, reader.cols, CV_32FC1, raw_disp.get());
        }

        cv::Size resize_to(left_img.size() / 4);
        cv::Mat disp_l2r_roi = disp_l2r;
        {
            cv::resize(left_img, left_img, resize_to);
            cv::resize(right_img, right_img, resize_to);
            cv::resize(disp_l2r, disp_l2r, resize_to);
            disp_l2r.convertTo(disp_l2r_roi, CV_8UC1);
        }

        // cv::Rect roi(1500, 700, 5, 5);
        // cv::Mat disp_l2r_roi = disp_l2r(roi);

        cv::Mat left, right;
        // cv::cvtColor(left_img(roi), left, cv::COLOR_BGR2GRAY);
        // cv::cvtColor(right_img(roi), right, cv::COLOR_BGR2GRAY);
        cv::cvtColor(left_img, left, cv::COLOR_BGR2GRAY);
        cv::cvtColor(right_img, right, cv::COLOR_BGR2GRAY);

        // test code:
        cv::Mat cpp_l2r;
        {
            int w_size = 5;
            (void)w_size;

            int max_disp = 50;
            cv::Mat left_mean = test_make_mean(left, w_size);
            cv::Mat right_mean = test_make_mean(right, w_size);
#if 0
            cpp_l2r = make_disparity_map(left, left_mean, right, right_mean, w_size, max_disp);
            // fill_occlusions_disparity(cpp_l2r, w_size);
#else
            cv::Mat cpp_l2r_ = stereo_compute_disparity(left, right, w_size, max_disp);
            cpp_l2r = cpp_l2r_;
#endif
        }

        // PRINTLN(cpp_l2r);
        // PRINTLN(disp_l2r_roi);

        // cv::Size resize_show(resize_to / 2);
        // {
        // cv::resize(disp_l2r_roi, disp_l2r_roi, resize_show);
        // cv::resize(cpp_l2r, cpp_l2r, resize_show);
        // }

        cv::imshow("GT disparity L2R", disp_l2r_roi);
        cv::imshow("Computed disparity L2R", cpp_l2r);
        cv::waitKey();

        // REQUIRE(cv::countNonZero(disp_l2r_roi != cpp_l2r) == 0);
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
