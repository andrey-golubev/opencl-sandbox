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

// OpenCL can produce inaccurate results (compared to C++), so comparing with tolerance
bool equal_with_tolerance(cv::Mat in1, cv::Mat in2, double abs_tolerance) {
    double err = cv::norm(in1, in2, cv::NORM_INF);
    double tolerance = abs_tolerance;
    if (err > tolerance) {
        std::cout << "equality check fail: err=" << err << ", accepted tolerance=" << tolerance
                  << std::endl;
        return false;
    } else {
        return true;
    }
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

    cv::Mat ocv = process_rgb_cpp(rgb);
    cv::Mat ocl = process_rgb_ocl(rgb);

    // TODO: might be useful for debugging
    // auto gray = rgb2gray_cpp(rgb);
    // PRINTLN("----");
    // PRINTLN(gray);
    // PRINTLN("----");
    // PRINTLN(moving_avg_cpp(gray));
    // PRINTLN("----");
    // PRINTLN(moving_avg_ocl(gray));
    // PRINTLN("----");

    auto processed_name_ocv = parts[0] + "_rgb2gray_blur_ocv." + parts[1];
    PRINTLN("Saved processed image to: " + processed_name_ocv);
    cv::imwrite(processed_name_ocv, ocv);

    auto processed_name_ocl = parts[0] + "_rgb2gray_blur_ocl." + parts[1];
    PRINTLN("Saved processed image to: " + processed_name_ocl);
    cv::imwrite(processed_name_ocl, ocl);

    // TODO: not always consistent (OpenCL specifics?)
    REQUIRE(equal_with_tolerance(ocv, ocl, 2));

    return 0;
}
