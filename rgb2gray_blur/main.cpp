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

namespace {
void print_usage(const char* program_name) {
    PRINTLN("Usage: " + std::string(program_name) + " IMAGE [CL_PLATFORM_ID CL_DEVICE_ID]");
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
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }
    if (argc != 2 && argc != 4) {
        print_usage(argv[0]);
        return 1;
    }

    int platform_id = 0, device_id = 0;
    if (argc == 4) {
        platform_id = std::stoi(argv[2]);
        device_id = std::stoi(argv[3]);
    }

    auto parts = split(std::string(argv[1]), '.');
    if (parts.size() != 2) {
        PRINTLN("Error: input image name is incorrect");
        return 1;
    }

    PRINTLN("-----");
    PRINTLN("For OpenCL: using platform #" + std::to_string(platform_id) + " and device #" +
            std::to_string(device_id));
    PRINTLN("-----\n");

    cv::Mat bgr = cv::imread(argv[1]);  // OpenCV always reads in BGR instead of RGB
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat ocv = process(rgb);
    auto processed_name = parts[0] + "_gray_blur." + parts[1];
    PRINTLN("Saved processed image to: " + processed_name);
    cv::imwrite(processed_name, ocv);

    return 0;
}
