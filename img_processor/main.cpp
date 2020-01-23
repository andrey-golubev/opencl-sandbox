#include <cstdint>
#include <iostream>

#include <opencv2/imgcodecs.hpp>

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

uint8_t rgb2gray(uint8_t r, uint8_t g, uint8_t b) { return 0.3 * r + 0.59 * g + 0.11 * b; }

uint8_t threshold(uint8_t pixel, uint8_t thr) {
    if (pixel < thr) {
        return 0;
    }
    return pixel;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " IMAGE" << std::endl;
        return 1;
    }
    auto parts = split(std::string(argv[1]), '.');
    if (parts.size() != 2) {
        std::cout << "Error: input image name is incorrect" << std::endl;
        return 1;
    }

    cv::Mat bgr = cv::imread(argv[1]);

    cv::Mat processed(bgr.size(), CV_8UC1);
    for (int i = 0; i < bgr.rows; ++i) {
        for (int j = 0; j < bgr.cols; ++j) {
            auto bgr_pixel = bgr.at<cv::Vec3b>(i * bgr.rows + j);
            processed.at<uint8_t>(i * bgr.rows + j) =
                threshold(rgb2gray(bgr_pixel[2], bgr_pixel[1], bgr_pixel[0]), 128);
        }
    }

    cv::imwrite(parts[0] + "_gray_thr." + parts[1], processed);

    return 0;
}
