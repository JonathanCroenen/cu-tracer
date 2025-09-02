#include "common.cuh"
#include "framebuffer.cuh"
#include "kernels/path_tracing.cuh"
#include "kernels/trace_attribute.cuh"
#include "renderer.cuh"

namespace rt {

KERNEL void InitRandomStatesKernel(curandState* random_states, int num_states,
                                   unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(seed, idx, 0, &random_states[idx]);
    }
}

bool Renderer::Init() {
    CUDA_CHECK(cudaMalloc(&_device_random_states, width * height * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&_device_accumulator, width * height * sizeof(Vec3f)));

    InitRandomStatesKernel<<<(width * height + 255) / 256, 256>>>(_device_random_states,
                                                                  width * height, time(nullptr));

    sample_count = 0;
    return true;
}

void Renderer::Cleanup() {
    if (_device_random_states) {
        cudaFree(_device_random_states);
        _device_random_states = nullptr;
    }

    if (_device_accumulator) {
        cudaFree(_device_accumulator);
        _device_accumulator = nullptr;
    }
}

void Renderer::ClearAccumulator() {
    CUDA_CHECK(cudaMemset(_device_accumulator, 0, width * height * sizeof(Vec3f)));
    sample_count = 0;
}

void Renderer::RenderFrame(Framebuffer& framebuffer, const Scene& scene, RenderMode mode) {
    sample_count++;
    auto surface = framebuffer.Map();

    // Launch ray tracing kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    switch (mode) {
    case RenderMode::PATH_TRACING:
        PathTracingKernel<<<gridSize, blockSize>>>(surface, width, height, scene, rays_per_pixel,
                                                   max_bounces, _device_random_states,
                                                   _device_accumulator, sample_count);
        break;
    case RenderMode::DEBUG_NORMALS:
        TraceAttributeDebugKernel<<<gridSize, blockSize>>>(surface, width, height, scene,
                                                           TraceAttribute::NORMALS);
        break;
    case RenderMode::DEBUG_DEPTH:
        TraceAttributeDebugKernel<<<gridSize, blockSize>>>(surface, width, height, scene,
                                                           TraceAttribute::DEPTH);
        break;
    }

    framebuffer.Unmap();
    CUDA_CHECK(cudaStreamSynchronize(0));
}

}  // namespace rt
