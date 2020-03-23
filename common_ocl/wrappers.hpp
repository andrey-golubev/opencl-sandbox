#pragma once

#include "common_ocl/utils.hpp"

struct OclMem {
    cl_mem memory = nullptr;

    OclMem() = default;
    OclMem(const OclMem&) = delete;
    OclMem& operator=(const OclMem&) = delete;
    OclMem(OclMem&& other) { OCL_MOVE_PTR(memory, other.memory); }
    OclMem& operator=(OclMem&& other) {
        OCL_MOVE_PTR(memory, other.memory);
        return *this;
    }

    OclMem(cl_mem mem) : memory(mem) {}
    OclMem& operator=(cl_mem mem) {
        memory = mem;
        return *this;
    }

    ~OclMem() {
        if (memory == nullptr) {
            return;
        }
        OCL_GUARD(clReleaseMemObject(memory));
    }

    operator cl_mem() { return memory; }
    operator cl_mem*() { return &memory; }
};