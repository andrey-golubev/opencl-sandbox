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

namespace {
void print_usage(const char* program_name) {
    PRINTLN("Usage: " + std::string(program_name) + " IMAGE_LEFT IMAGE_RIGHT [MAX_DISPARITY]");
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
}  // namespace

// debug controls
#define OPTIMIZED 0
#define SHOW_WINDOW 1

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    // read input image
    cv::Mat bgr_left = cv::imread(argv[1]);
    cv::Mat bgr_right = cv::imread(argv[2]);

    int max_disparity = 50;

    // read disparity from user input if specified
    if (argc > 3) {
        max_disparity = std::stoi(argv[3]);
    }

    // convert to grayscale
    cv::Mat left;
    cv::Mat right;
    cv::cvtColor(bgr_left, left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgr_right, right, cv::COLOR_BGR2GRAY);

    // find disparity
#if OPTIMIZED
    cv::Mat map = stereo_cpp_opt::stereo_compute_disparity(left, right, max_disparity);
#else
    cv::Mat map = stereo_cpp_base::stereo_compute_disparity(left, right, max_disparity);
#endif

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
