#include <cuda_gl_interop.h>
#include "common.cuh"
#include "framebuffer.cuh"
#include "renderer.cuh"

namespace rt {

DEVICE Vec3f TraceRay(const Rayf& ray, const Scene& scene, int max_bounces,
                      curandState* random_state);

KERNEL void InitRandomStatesKernel(curandState* random_states, int num_states,
                                   unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_states) {
        curand_init(seed, idx, 0, &random_states[idx]);
    }
}

KERNEL void RenderKernel(cudaSurfaceObject_t surface, int width, int height, const Scene& scene,
                         int rays_per_pixel, int max_bounces, curandState* random_states,
                         Vec3f* accumulated_colors, int sample_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    curandState* local_state = &random_states[y * width + x];

    Vec3f color = Vec3f(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < rays_per_pixel; i++) {
        float u = (float(x) + RandomFloat(local_state)) / float(width - 1);
        float v = (float(y) + RandomFloat(local_state)) / float(height - 1);

        Rayf ray = scene.GetCameraRay(u, v);
        color += TraceRay(ray, scene, max_bounces, local_state);
    }

    color /= float(rays_per_pixel);

    int pixel_index = y * width + x;
    accumulated_colors[pixel_index] = accumulated_colors[pixel_index] + color;
    Vec3f average_color = accumulated_colors[pixel_index] / float(sample_count);

    surf2Dwrite(make_float4(average_color.x, average_color.y, average_color.z, 1.0f), surface,
                x * sizeof(float4), y);
}

DEVICE Vec3f TraceRay(const Rayf& initial_ray, const Scene& scene, int max_bounces,
                      curandState* random_state) {
    Rayf ray = initial_ray;
    Vec3f throughput(1.0f, 1.0f, 1.0f);

    for (int bounce = 0; bounce < max_bounces; ++bounce) {
        Hit hit;
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
            // return Vec3f(0.0f, 0.0f, 0.0f);
        }
    }

    return Vec3f(0, 0, 0);
}

Renderer::Renderer(int w, int h, int rays_per_pixel, int max_bounces)
    : width(w), height(h), rays_per_pixel(rays_per_pixel), max_bounces(max_bounces),
      sample_count(0), _device_random_states(nullptr), _device_accumulator(nullptr) {}

Renderer::~Renderer() {
    Cleanup();
}

bool Renderer::Init() {
    CUDA_CHECK(cudaMalloc(&_device_random_states, width * height * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&_device_accumulator, width * height * sizeof(Vec3f)));

    InitRandomStates();

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

void Renderer::InitRandomStates() {
    InitRandomStatesKernel<<<(width * height + 255) / 256, 256>>>(_device_random_states,
                                                                  width * height, time(nullptr));
}

void Renderer::ClearAccumulator() {
    CUDA_CHECK(cudaMemset(_device_accumulator, 0, width * height * sizeof(Vec3f)));
    sample_count = 0;
}

void Renderer::RenderFrame(Framebuffer& framebuffer, const Scene& scene) {
    sample_count++;
    framebuffer.Map();

    // Launch ray tracing kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    RenderKernel<<<gridSize, blockSize>>>(framebuffer.GetSurface(), width, height, scene,
                                          rays_per_pixel, max_bounces, _device_random_states,
                                          _device_accumulator, sample_count);

    framebuffer.Unmap();
    CUDA_CHECK(cudaStreamSynchronize(0));
}

}  // namespace rt
