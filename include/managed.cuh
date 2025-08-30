#pragma once

#include <cuda_runtime.h>

struct Managed {
    void* operator new(size_t size) {
        void* ptr;
        cudaMallocManaged(&ptr, size);
        return ptr;
    }

    void operator delete(void* ptr) {
        cudaFree(ptr);
    }
};