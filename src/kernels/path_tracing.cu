#include "kernels/path_tracing.cuh"

namespace rt {

DEVICE Vec3f TraceRay(const Rayf& initial_ray, const Scene& scene, int max_bounces,
                      curandState* random_state) {
    Rayf ray = initial_ray;
    Vec3f throughput(1.0f, 1.0f, 1.0f);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        HitRecord hit;
        if (scene.Hit(ray, hit)) {
            Rayf scattered;
            Vec3f attenuation(0.0f);
            if (hit.material->Scatter(ray, hit, attenuation, scattered, random_state)) {
                ray = scattered;
                throughput *= attenuation;
                continue;
            }

            return throughput * attenuation;
        } else {
            float t = 0.5f * (ray.direction.y + 1.0f);
            Vec3f sky_color = Vec3f(0.7f, 0.7f, 0.7f) * (1.0f - t) + Vec3f(0.2f, 0.3f, 0.7f) * t;
            return throughput * sky_color;
        }
    }

    return Vec3f(0, 0, 0);
}

KERNEL void PathTracingKernel(cudaSurfaceObject_t surface, int width, int height,
                              const Scene& scene, int rays_per_pixel, int max_bounces,
                              curandState* random_states, Vec3f* accumulated_colors,
                              int sample_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    curandState* local_state = &random_states[y * width + x];

    Vec3f color = Vec3f(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < rays_per_pixel; i++) {
        float u = (float(x) + RandomFloat(local_state)) / float(width);
        float v = (float(y) + RandomFloat(local_state)) / float(height);

        Rayf ray = scene.GetCameraRay(u, v);
        color += TraceRay(ray, scene, max_bounces, local_state);
    }

    color /= float(rays_per_pixel);
    color = Vec3f(sqrt(color.x), sqrt(color.y), sqrt(color.z));
    color = color.Clamped(0.0f, 1.0f);

    int pixel_index = y * width + x;
    accumulated_colors[pixel_index] += color;
    Vec3f average_color = accumulated_colors[pixel_index] / float(sample_count);

    surf2Dwrite(make_float4(average_color.x, average_color.y, average_color.z, 1.0f), surface,
                x * sizeof(float4), y);
}

}  // namespace rt