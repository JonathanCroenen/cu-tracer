#pragma once

#include <iostream>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                    \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

#define DEVICE __device__
#define HOST __host__
#define DEVICE_HOST __device__ __host__
#define KERNEL __global__
