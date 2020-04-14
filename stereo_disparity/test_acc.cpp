#include <cstdint>
#include <iostream>
#include <map>
#include <random>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>

// CPP stereo algo:
#include "stereo_disparity_cpp_inl.hpp"
// CPP optimized stereo algo:
#include "stereo_disparity_cpp_opt_inl.hpp"
// OCL stereo algo:
#include "stereo_disparity_ocl_inl.hpp"

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
    cv::Size(1920, 1080), cv::Size(640, 480), cv::Size(189, 279),
    cv::Size(12, 12),     cv::Size(200, 5),   cv::Size(5, 5),
};

const cv::Size DISPARITY_TEST_SIZES[] = {
    cv::Size(320, 240), cv::Size(99, 99), cv::Size(137, 80), cv::Size(77, 16), cv::Size(59, 6),
};

cv::Mat test_make_mean(const cv::Mat& in, int k_size) {
    cv::Mat out;
    cv::boxFilter(in, out, -1, cv::Size(k_size, k_size));
    return out;
}

std::string TEST_DATA_FOLDER = "";

void declare_tests() {
    TEST(INDEX_FIX_FUNCTION) {
        REQUIRE(stereo_common::fix(9, 10, 12) == 11);
        REQUIRE(stereo_common::fix(-1, 0, 5) == 1);
        REQUIRE(stereo_common::fix(6, 7, 9) == 8);
        REQUIRE(stereo_common::fix(13, 10, 12) == 11);
    };
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
    TEST(CONVOLVE) {
        const cv::Size sizes[] = {
            cv::Size(1920, 1080), cv::Size(640, 480), cv::Size(189, 279),
            cv::Size(16, 16),     cv::Size(200, 5),
        };
        for (const auto& size : sizes) {
            cv::Mat in(size, CV_8UC1);
            cv::randu(in, cv::Scalar(0), cv::Scalar(255));

            for (int k_size : {3, 5, 9}) {
                cv::Mat kernel = cv::Mat::zeros(cv::Size(k_size, k_size), CV_8UC1);
                cv::randu(kernel, cv::Scalar::all(0), cv::Scalar::all(10));
                cv::Mat float_kernel;
                kernel.convertTo(float_kernel, CV_32FC1);

                cv::Mat ocv_tmp, cpp;
                cv::filter2D(in, ocv_tmp, CV_32FC1, float_kernel, cv::Point(-1, -1), 0.0,
                             cv::BORDER_CONSTANT);
                cv::Mat ocv;
                ocv_tmp.convertTo(ocv, CV_32SC1);

                cpp = stereo_cpp_opt::mat_conv(in, kernel, 1);
                REQUIRE(cv::countNonZero(ocv != cpp) == 0);
            }
        }
    };
    TEST(COPY_MAKE_BORDER_OPT) {
        const cv::Size sizes[] = {cv::Size(1920, 1080), cv::Size(640, 480), cv::Size(189, 279),
                                  cv::Size(16, 16), cv::Size(200, 6)};
        for (const auto& size : sizes) {
            cv::Mat in(size, CV_8UC1);
            cv::randu(in, cv::Scalar(0), cv::Scalar(255));

            cv::Mat ocv;

            for (int r_border : {0, 1, 2, 3, 4, 5}) {
                for (int c_border : {0, 1, 2, 3, 4, 5}) {
                    cv::copyMakeBorder(in, ocv, r_border, r_border, c_border, c_border,
                                       cv::BORDER_REFLECT101);
                    cv::Mat cpp = stereo_cpp_opt::copy_make_border(in, r_border, c_border);
                    REQUIRE(cv::countNonZero(ocv != cpp) == 0);
                }
            }
        }
    };
    TEST(COPY_LINE_BORDER_OPT) {
        const cv::Size sizes[] = {cv::Size(1920, 1080), cv::Size(640, 480), cv::Size(189, 279),
                                  cv::Size(16, 16)};
        for (const auto& size : sizes) {
            cv::Mat in(size, CV_8UC1);
            cv::randu(in, cv::Scalar(0), cv::Scalar(255));

            cv::Mat ocv;

            for (int y_shift : {0, 1, 2, 3, 4, 5}) {
                for (int height_shift : {0, 1, 2, 3, 4, 5}) {
                    for (int c_border : {0, 1, 2, 3, 4, 5}) {
                        detail::HorSlice slice{y_shift, size.height - height_shift - y_shift};
                        cv::Rect roi(0, slice.y, size.width, slice.height);
                        ocv = in(roi);
                        cv::copyMakeBorder(ocv, ocv, 0, 0, c_border, c_border,
                                           cv::BORDER_REFLECT101);
                        cv::Mat cpp = stereo_cpp_opt::copy_line_border(in, slice, c_border);
                        REQUIRE(cv::countNonZero(ocv != cpp) == 0);
                    }
                }
            }
        }
    };
    TEST(DISPARITY_MAP_SANITY) {
#define SHOW 0
#if SHOW
        for (auto append : {"/backpack/"}) {
#else
        for (auto append : {"/backpack/", "/umbrella/"}) {
#endif
            std::string folder = TEST_DATA_FOLDER + append;
            cv::Mat left_img = cv::imread(folder + "im0.png");
            cv::Mat right_img = cv::imread(folder + "im1.png");

#if SHOW
            cv::Size sz(640, 480);  // resized down for increased speed
#else
            cv::Size sz(160, 120);  // resized down for increased speed
#endif
            {
                cv::resize(left_img, left_img, sz);
                cv::resize(right_img, right_img, sz);
            }

            cv::Mat left, right;
            cv::cvtColor(left_img, left, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right_img, right, cv::COLOR_BGR2GRAY);

            cv::Mat cpp_disp;

#if SHOW
            int max_disp = 200;
#else
            int max_disp = 50;
#endif
            cpp_disp = stereo_cpp_opt::stereo_compute_disparity(left, right, max_disp);

            cv::Mat zero = cv::Mat::zeros(cpp_disp.size(), cpp_disp.type());

            REQUIRE(cpp_disp.type() == CV_8UC1);
            REQUIRE(cv::countNonZero(zero == cpp_disp) < (cpp_disp.rows * cpp_disp.cols));

#if SHOW
            cv::imshow("DISP", cpp_disp);
#endif
        }
#if SHOW
        while (cv::waitKey() != 27)
            ;
#endif
    };

    TEST(DISPARITY_MAP_OPT) {
        for (auto append : {"/backpack/", "/umbrella/"}) {
            std::string folder = TEST_DATA_FOLDER + append;
            cv::Mat left_img = cv::imread(folder + "im0.png");
            cv::Mat right_img = cv::imread(folder + "im1.png");

            for (auto sz : DISPARITY_TEST_SIZES) {
                {
                    cv::resize(left_img, left_img, sz);
                    cv::resize(right_img, right_img, sz);
                }

                cv::Mat left, right;
                cv::cvtColor(left_img, left, cv::COLOR_BGR2GRAY);
                cv::cvtColor(right_img, right, cv::COLOR_BGR2GRAY);

                int max_disp = 50;
                cv::Mat base_disp, opt_disp;

                opt_disp = stereo_cpp_opt::stereo_compute_disparity(left, right, max_disp);
                base_disp = stereo_cpp_base::stereo_compute_disparity(left, right, max_disp);

                REQUIRE(base_disp.type() == opt_disp.type());
                REQUIRE(base_disp.size() == opt_disp.size());

                REQUIRE(cv::countNonZero(base_disp != opt_disp) == 0);
            }
        }
    };

    TEST(DISPARITY_MAP_OCL) {
        for (auto append : {"/backpack/", "/umbrella/"}) {
            std::string folder = TEST_DATA_FOLDER + append;
            cv::Mat left_img = cv::imread(folder + "im0.png");
            cv::Mat right_img = cv::imread(folder + "im1.png");

            for (auto sz : DISPARITY_TEST_SIZES) {
                {
                    cv::resize(left_img, left_img, sz);
                    cv::resize(right_img, right_img, sz);
                }

                cv::Mat left, right;
                cv::cvtColor(left_img, left, cv::COLOR_BGR2GRAY);
                cv::cvtColor(right_img, right, cv::COLOR_BGR2GRAY);

                int max_disp = 50;
                cv::Mat cpp_disp, ocl_disp;

                ocl_disp = stereo_ocl_base::stereo_compute_disparity(left, right, max_disp,
                                                                     platform_id, device_id);
                cpp_disp = stereo_cpp_base::stereo_compute_disparity(left, right, max_disp);

                REQUIRE(cpp_disp.type() == ocl_disp.type());
                REQUIRE(cpp_disp.size() == ocl_disp.size());

                REQUIRE(cv::countNonZero(cpp_disp != ocl_disp) == 0);
            }
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
