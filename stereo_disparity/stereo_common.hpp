#pragma once

#include <opencv2/core.hpp>

namespace stereo_common {
// constants:
constexpr const double EPS = 0.0005;
constexpr const uchar UNKNOWN_DISPARITY = 0;
constexpr const int WINDOW_SIZE = 11;
constexpr const double ZNCC_THRESHOLD = 0.995;
}  // namespace stereo_common
