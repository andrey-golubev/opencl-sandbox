#pragma once

#include <opencv2/core.hpp>

namespace stereo_common {
// constants:
constexpr const double EPS = 0.0005;
constexpr const uchar UNKNOWN_DISPARITY = 0;
constexpr const int MAX_WINDOW = 11;
constexpr const int MAX_BORDER = (MAX_WINDOW - 1) / 2;
}  // namespace stereo_common
