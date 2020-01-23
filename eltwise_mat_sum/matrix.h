#pragma once

#include <array>
#include <ostream>
#include <tuple>
#include <vector>

template<typename Type> struct Matrix {
    std::vector<std::vector<Type>> data = {};
};

using RGBCol = std::vector<std::array<uint8_t, 3>>;
using RGBMat = Matrix<std::array<uint8_t, 3>>;

static std::ostream& operator<<(std::ostream& os, const RGBMat& mat) {
    for (const auto& row : mat.data) {
        for (const auto& elem : row) {
            os << "(" << std::to_string(elem[0]);
            for (size_t chan = 1; chan < elem.size(); ++chan) {
                os << " " << std::to_string(elem[chan]);
            }
            os << ") ";
        }
        os << std::endl;
    }
    return os;
}
