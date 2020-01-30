#pragma once

#include <cstdint>
#include <stdexcept>

#include <opencv2/core.hpp>

// interface:
cv::Mat eltwise_add_cpp(const cv::Mat& a, const cv::Mat& b) { return a + b; }