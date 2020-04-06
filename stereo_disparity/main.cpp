#include <cstdint>
#include <iostream>
#include <map>
#include <random>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/utils.hpp"

#include "stereo_disparity_cpp_inl.hpp"
#include "stereo_disparity_cpp_opt_inl.hpp"
#include "stereo_disparity_ocl_inl.hpp"

namespace {
void print_usage(const char* program_name) {
    PRINTLN("Usage: " + std::string(program_name) +
            " IMAGE_LEFT IMAGE_RIGHT [ALGO_VERSION] [MAX_DISPARITY]\n" +
            "  ALGO_VERSION:\n  0 - C++ basic\n  1 - C++ optimized\n  2 - OpenCL");
}

template<typename CharT, typename Traits = std::char_traits<CharT>>
std::vector<std::basic_string<CharT>> split(const std::basic_string<CharT>& src, CharT delimiter) {
    std::vector<std::basic_string<CharT>> dst;
    std::basic_istringstream<CharT> ss_src(src);
    std::basic_string<CharT> tmp;
    while (std::getline(ss_src, tmp, delimiter)) {
        dst.push_back(tmp);
    }
    return dst;
}

enum AlgoType {
    CPP_BASIC = 0,
    CPP_OPT = 1,
    OCL = 2,
};

std::string algo2str(int t) {
    switch (t) {
    case CPP_BASIC:
        return "C++ basic";
    case CPP_OPT:
        return "C++ optimized";
    case OCL:
        return "OpenCL";
    default:
        throw std::runtime_error("Unknown algorithm version");
    }
}
}  // namespace

// debug controls
#define SHOW_WINDOW 0
#define RESIZE 0

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        print_usage(argv[0]);
        return 1;
    }

    // read input image
    cv::Mat bgr_left = cv::imread(argv[1]);
    cv::Mat bgr_right = cv::imread(argv[2]);

    int algo_version = 0;
    int max_disparity = 50;

    // read disparity from user input if specified
    if (argc > 3) {
        if (argc >= 4) {
            algo_version = std::stoi(argv[3]);
        }
        if (argc == 5) {
            max_disparity = std::stoi(argv[4]);
        }
    }

    // convert to grayscale
    cv::Mat left;
    cv::Mat right;
    cv::cvtColor(bgr_left, left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgr_right, right, cv::COLOR_BGR2GRAY);

#if RESIZE
    const auto input_size = left.size();
    constexpr double scale = 0.5;
    auto size = cv::Size(input_size.width * 0.5, input_size.height * scale);
    cv::resize(left, left, size);
    cv::resize(right, right, size);
#endif

    // find disparity
    cv::Mat map;
    uint64_t musec = 0;
    PRINTLN("Running " + algo2str(algo_version) + " version");
    switch (algo_version) {
    case CPP_BASIC: {
        musec = measure(
            1,
            [&]() { map = stereo_cpp_base::stereo_compute_disparity(left, right, max_disparity); },
            false);
        break;
    }
    case CPP_OPT: {
        musec = measure(
            1,
            [&]() { map = stereo_cpp_opt::stereo_compute_disparity(left, right, max_disparity); },
            false);
        break;
    }
    case OCL: {
        musec = measure(
            1,
            [&]() { map = stereo_ocl_base::stereo_compute_disparity(left, right, max_disparity); },
            false);
        break;
    }
    default:
        throw std::runtime_error("Unknown algorithm version");
    }

    OUT << "Time: " << musec << " musec" << std::endl;

#if SHOW_WINDOW
    // show disparity map
    cv::String win_name("Disparity Map");
    cv::namedWindow(win_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(win_name, cv::Size(640 * 2, 480 * 2));
    cv::imshow(win_name, map);
    while (cv::waitKey(1) != 27) {
    };
#endif

    return 0;
}
