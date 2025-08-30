#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "framebuffer.cuh"
#include "math.cuh"
#include "scene.cuh"

namespace rt {

class Renderer {
public:
    int width;
    int height;
    int rays_per_pixel;
    int max_bounces;
    int sample_count;

    Renderer(int width, int height, int max_bounces, int rays_per_pixel);
    ~Renderer();

    bool Init();
    void Cleanup();
    void RenderFrame(Framebuffer& framebuffer, const Scene& scene);
    void ClearAccumulator();

private:
    curandState* _device_random_states;
    Vec3f* _device_accumulator;

    void InitRandomStates();
};

}  // namespace rt
