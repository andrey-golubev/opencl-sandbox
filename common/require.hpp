#pragma once

#include <stdexcept>

#define REQUIRE(expr)                                                                              \
    do {                                                                                           \
        if (!(expr))                                                                               \
            throw std::runtime_error(#expr);                                                       \
    } while (0)

#define REQUIRE2(expr, expr_str)                                                                   \
    do {                                                                                           \
        if (!(expr))                                                                               \
            throw std::runtime_error((expr_str));                                                  \
    } while (0)

#define REQUIRE_CUSTOM(expr, expr_str, handle_error)                                               \
    do {                                                                                           \
        if (!(expr))                                                                               \
            throw std::runtime_error((handle_error((expr_str))));                                  \
    } while (0)
