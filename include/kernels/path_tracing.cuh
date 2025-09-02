#pragma once

#include "../common.cuh"
#include "../math.cuh"
#include "../scene.cuh"

namespace rt {

KERNEL void PathTracingKernel(cudaSurfaceObject_t surface, int width, int height,
                              const Scene& scene, int rays_per_pixel, int max_bounces,
                              curandState* random_states, Vec3f* accumulated_colors,
                              int sample_count);

}  // namespace rt