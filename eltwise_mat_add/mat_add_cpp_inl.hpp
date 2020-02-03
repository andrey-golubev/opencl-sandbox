#pragma once

#include <cstdint>
#include <stdexcept>

#include <opencv2/core.hpp>

// interface:
cv::Mat eltwise_add_ocv(const cv::Mat& a, const cv::Mat& b) { return a + b; }

cv::Mat eltwise_add_cpp(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat out = cv::Mat::zeros(a.size(), a.type());

    int chan = a.channels();
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            for (int c = 0; c < chan; ++c) {
                int idx = i * a.cols * chan + j * chan + c;
                out.at<uchar>(idx) = a.at<uchar>(idx) + b.at<uchar>(idx);
            }
        }
    }

    return out;
}
