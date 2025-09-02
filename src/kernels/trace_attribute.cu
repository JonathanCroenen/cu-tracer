#include "kernels/trace_attribute.cuh"
#include "math.cuh"

namespace rt {

DEVICE Vec3f TraceRay(const Rayf& ray, const Scene& scene, TraceAttribute attribute) {
    HitRecord hit;
    if (scene.Hit(ray, hit)) {
        switch (attribute) {
        case TraceAttribute::NORMALS: return hit.normal * 0.5f + 0.5f;
        case TraceAttribute::DEPTH: {
            float mapped_depth = 0.5f * hit.t / (1.0f + hit.t);
            return Vec3f(mapped_depth);
        }
        }
    }
    return Vec3f();
}

KERNEL void TraceAttributeDebugKernel(cudaSurfaceObject_t surface, int width, int height,
                                      const Scene& scene, TraceAttribute attribute) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float u = float(x) / float(width);
    float v = float(y) / float(height);

    Rayf ray = scene.GetCameraRay(u, v);
    Vec3f value = TraceRay(ray, scene, attribute);

    surf2Dwrite(make_float4(value.x, value.y, value.z, 1.0f), surface, x * sizeof(float4), y);
}

}  // namespace rt