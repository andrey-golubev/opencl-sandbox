#include <cstdint>
#include <iostream>
#include <random>

// CPP version of matrix summation:
#include "mat_sum_cpp_inl.hpp"
// OpenCL version of matrix summation:
#include "mat_sum_ocl_inl.hpp"

namespace {
RGBMat generateRandom(size_t rows, size_t cols) {
    RGBMat mat;
    mat.data.resize(rows, RGBCol(cols));

    static std::mt19937 gen;

    std::uniform_int_distribution<uint8_t> dist(0, 100);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            auto& rgb = mat.data[i][j];

            for (size_t chan = 0; chan < rgb.size(); ++chan) {
                rgb[chan] = dist(gen);
            }
        }
    }
    return mat;
}

bool operator==(const RGBMat& a, const RGBMat& b) {
    size_t rows = a.data.size(), cols = a.data[0].size();
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            const auto& a_pixel = a.data[i][j];
            const auto& b_pixel = b.data[i][j];
            for (size_t chan = 0; chan < a_pixel.size(); ++chan) {
                if (a_pixel[chan] != b_pixel[chan]) {
                    return false;
                }
            }
        }
    }
    return true;
}
}  // namespace

int main(int argc, char* argv[]) {
    constexpr const size_t ROWS = 2;
    constexpr const size_t COLS = 2;

    auto a = generateRandom(ROWS, COLS);
    auto b = generateRandom(ROWS, COLS);
    std::cout << a << std::endl;
    std::cout << b << std::endl;

    std::cout << eltwise_sum_cpp(a, b) << std::endl;
    return 0;
}
