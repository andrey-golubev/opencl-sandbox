#include <cstdint>
#include <iostream>
#include <map>
#include <random>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "common/utils.hpp"

namespace {
void print_usage(const char* program_name) {
    PRINTLN("Usage: " + std::string(program_name) +
            " IMAGE_LEFT IMAGE_RIGHT [CL_PLATFORM_ID CL_DEVICE_ID]");
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

int main(int argc, char* argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    if (argc != 3 && argc != 5) {
        print_usage(argv[0]);
        return 1;
    }

    int platform_id = 0, device_id = 0;
    if (argc == 5) {
        platform_id = std::stoi(argv[2]);
        device_id = std::stoi(argv[3]);
    }

    PRINTLN("-----");
    PRINTLN("For OpenCL: using platform #" + std::to_string(platform_id) + " and device #" +
            std::to_string(device_id));
    PRINTLN("-----\n");

    // Read input image:
    cv::Mat bgr_left = cv::imread(argv[1]);
    cv::Mat bgr_right = cv::imread(argv[2]);

    // Convert to grayscale
    cv::Mat left;
    cv::Mat right;
    cv::cvtColor(bgr_left, left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgr_right, right, cv::COLOR_BGR2GRAY);

    return 0;
}
