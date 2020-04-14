#pragma once

#define NONE 0
#define OPENMP 1
#define SIMPLE 2
#define ADVANCE 3

#ifndef THREADS
#define THREADS NONE
#endif

namespace thr {
#if THREADS == NONE
template<typename T, typename Callable> inline void parallel_for(T iters, Callable f) {
    for (T i = 0; i < iters; ++i) {
        f(i, iters);
    }
}
inline int get_max_threads() { return 1; }
inline void set_num_threads(int n) { return; }
#elif THREADS == OPENMP
#include <omp.h>
template<typename T, typename Callable> void parallel_for(T iters, Callable f) {
    if (iters == 1) {
        f(0, 1);
        return;
    }

#pragma omp parallel num_threads(iters)
    f(omp_get_thread_num(), omp_get_num_threads());
}
inline int get_max_threads() { return omp_get_max_threads(); }
inline void set_num_threads(int n) { omp_set_num_threads(n); }

// TODO: implement other threading backends
#elif THREADS == SIMPLE
template<typename T, typename Callable> inline void parallel_for(T iters, Callable f) {
    for (T i = 0; i < iters; ++i) {
        f(i, iters);
    }
}
inline int get_max_threads() { return 1; }
inline void set_num_threads(int n) { return; }
#elif THREADS == ADVANCE
template<typename T, typename Callable> inline void parallel_for(T iters, Callable f) {
    for (T i = 0; i < iters; ++i) {
        f(i, iters);
    }
}
inline int get_max_threads() { return 1; }
inline void set_num_threads(int n) { return; }
#endif
}  // namespace thr
