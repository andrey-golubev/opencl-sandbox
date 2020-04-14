#pragma once

#define NONE 0
#define OPENMP 1
#define SIMPLE 2
#define ADVANCE 3

#ifndef THREADS
#define THREADS NONE
#endif

#if THREADS == OPENMP
#include <omp.h>
#elif THREADS == SIMPLE
#include <thread>
#include <vector>
#elif THREADS == ADVANCE
#include "thread_pool.hpp"
#endif

namespace thr {
#if THREADS == NONE
template<typename T, typename Callable> inline void parallel_for(T iters, Callable f) {
    for (T i = 0; i < iters; ++i) {
        f(i, iters);
    }
}
inline int get_max_threads() { return 1; }
#elif THREADS == OPENMP
template<typename T, typename Callable> void parallel_for(T iters, Callable f) {
    if (iters == 1) {
        f(0, 1);
        return;
    }

#pragma omp parallel num_threads(iters)
    f(omp_get_thread_num(), omp_get_num_threads());
}
inline int get_max_threads() { return omp_get_max_threads(); }

#elif THREADS == SIMPLE
template<typename T, typename Callable> inline void parallel_for(T iters, Callable f) {
    std::vector<std::thread> threads;
    threads.reserve(iters);

    for (T i = 0; i < iters; ++i) {
        threads.emplace_back(f, i, iters);
    }

    for (auto& t : threads) {
        t.join();
    }
}
inline int get_max_threads() { return std::thread::hardware_concurrency(); }
#elif THREADS == ADVANCE
inline int get_max_threads() { return std::thread::hardware_concurrency(); }

namespace detail {
inline ThreadPool& thread_pool() {
    static ThreadPool pool(std::thread::hardware_concurrency());
    return pool;
}
}  // namespace detail

template<typename T, typename Callable> inline void parallel_for(T iters, Callable f) {
    auto& pool = detail::thread_pool();

    std::vector<std::future<void>> promises{};
    promises.reserve(iters);

    for (T i = 0; i < iters; ++i) {
        promises.emplace_back(pool.add_task(f, i, iters));
    }

    for (auto& promise : promises) {
        promise.wait();
    }
}
#endif
}  // namespace thr
