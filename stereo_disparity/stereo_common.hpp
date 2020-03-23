#pragma once

#include <opencv2/core.hpp>

namespace stereo_common {
// constants:
constexpr const double EPS = 0.0005;
constexpr const uchar UNKNOWN_DISPARITY = 0;
constexpr const int MAX_WINDOW = 11;
constexpr const int MAX_BORDER = (MAX_WINDOW - 1) / 2;

int fix(int v, int min_v, int max_v) {
    if (v < min_v) {
        const int diff = (min_v - v);
        return min_v + diff;
    } else if (v > max_v) {
        const int diff = (v - max_v);
        return max_v - diff;
    }
    return v;
}

int fix(int v, int max_v) { return fix(v, 0, max_v); }
}  // namespace stereo_common
