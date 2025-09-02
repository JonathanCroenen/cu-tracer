#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "framebuffer.cuh"
#include "math.cuh"
#include "scene.cuh"

namespace rt {

enum class RenderMode {
    PATH_TRACING,
    DEBUG_NORMALS,
    DEBUG_DEPTH,
};

class Renderer {
public:
    int width;
    int height;
    int rays_per_pixel;
    int max_bounces;
    int sample_count;

    Renderer(int w, int h, int rays_per_pixel, int max_bounces)
        : width(w), height(h), rays_per_pixel(rays_per_pixel), max_bounces(max_bounces),
          sample_count(0), _device_random_states(nullptr), _device_accumulator(nullptr) {}

    ~Renderer() { Cleanup(); }

    bool Init();
    void Cleanup();
    void RenderFrame(Framebuffer& framebuffer, const Scene& scene,
                     RenderMode mode = RenderMode::PATH_TRACING);
    void ClearAccumulator();

private:
    curandState* _device_random_states;
    Vec3f* _device_accumulator;
};

}  // namespace rt
