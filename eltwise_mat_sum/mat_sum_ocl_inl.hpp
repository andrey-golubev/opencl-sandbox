#pragma once

#include "ocl_utils/ocl_utils.hpp"

#include "opencv2/core.hpp"

#include <algorithm>
#include <iostream>
#include <string>

#include <CL/cl.h>

// interface:
cv::Mat eltwise_sum_ocl(const cv::Mat& a, const cv::Mat& b) { return a + b; }
