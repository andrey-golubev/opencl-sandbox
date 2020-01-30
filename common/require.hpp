#pragma once

#include <stdexcept>

#define REQUIRE(expr)                                                                              \
    do {                                                                                           \
        if (!(expr))                                                                               \
            throw std::runtime_error(#expr);                                                       \
    } while (0)
