#pragma once

#include <opencv2/imgproc.hpp>

// interface:
cv::Mat process(cv::Mat in) {
    cv::Mat out = cv::Mat::zeros(in.size(), CV_8UC1);

    // 1. convert to grayscale
    cv::cvtColor(in, out, cv::COLOR_RGB2GRAY);

    // 2. apply 5x5 moving average

    // NB: moving average = convolution with 1's kernel + division by kernel size  a.k.a. blurring?
    cv::boxFilter(out, out, -1, cv::Size(5, 5), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    return out;
}
