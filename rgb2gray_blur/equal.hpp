#pragma once

#include <iostream>
#include <opencv2/core.hpp>

// OpenCL can produce inaccurate results (compared to C++), so comparing with tolerance
static bool equal_with_tolerance(cv::Mat in1, cv::Mat in2, double abs_tolerance) {
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
