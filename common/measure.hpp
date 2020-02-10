#pragma once

#include <chrono>
#include <cstdint>

template<typename IterFunction>
std::uint64_t measure(std::size_t iters, IterFunction f, bool warmup = true) {
    using namespace std::chrono;

    if (warmup) {
        f();  // warm-up
    }

    // measurement loop
    auto start = steady_clock::now();
    for (std::size_t _ = 0; _ < iters; ++_) {
        f();
    }
    auto end = steady_clock::now();

    return duration_cast<microseconds>(end - start).count();
}
