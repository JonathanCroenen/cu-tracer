#pragma once

#include "../common.cuh"
#include "../scene.cuh"

namespace rt {

enum class TraceAttribute {
    NORMALS,
    DEPTH,
};

KERNEL void TraceAttributeDebugKernel(cudaSurfaceObject_t surface, int width, int height,
                                      const Scene& scene, TraceAttribute attribute);

}  // namespace rt