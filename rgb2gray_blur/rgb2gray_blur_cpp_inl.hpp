#pragma once

#include <cstdint>

#include <opencv2/imgproc.hpp>

// individual operations:
cv::Mat rgb2gray_cpp(cv::Mat rgb);
cv::Mat moving_avg_cpp(cv::Mat gray);

// interface:
cv::Mat process_rgb_cpp(cv::Mat rgb) {
    // 1. convert to grayscale
    auto gray = rgb2gray_cpp(rgb);

    // 2. apply 5x5 moving average
    return moving_avg_cpp(gray);
}

uint8_t rgb2gray_impl(uint8_t r, uint8_t g, uint8_t b) { return 0.3 * r + 0.59 * g + 0.11 * b; }

cv::Mat rgb2gray_cpp(cv::Mat rgb) {
    cv::Mat out = cv::Mat::zeros(rgb.size(), CV_8UC1);

    for (int i = 0; i < rgb.rows; ++i) {
        for (int j = 0; j < rgb.cols; ++j) {
            auto rgb_pixel = rgb.at<cv::Vec3b>(i * rgb.cols + j);
            out.at<uint8_t>(i * rgb.cols + j) =
                rgb2gray_impl(rgb_pixel[0], rgb_pixel[1], rgb_pixel[2]);
        }
    }

    return out;
}

cv::Mat moving_avg_cpp(cv::Mat gray) {
    cv::Mat out(gray.size(), gray.type());
    // NB: moving average = convolution with 1's kernel + division by kernel size  a.k.a. blurring?
    cv::boxFilter(gray, out, -1, cv::Size(5, 5));
    return out;
}