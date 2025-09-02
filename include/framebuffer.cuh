#pragma once

namespace rt {

class Framebuffer {
public:
    Framebuffer() : _cuda_texture_resource(nullptr) {}
    Framebuffer(cudaGraphicsResource_t cuda_texture_resource)
        : _cuda_texture_resource(cuda_texture_resource) {}
    ~Framebuffer() {}

    cudaSurfaceObject_t Map() { return _GetMappedSurface(); }
    void Unmap() { _UnmapSurface(); }

private:
    cudaGraphicsResource_t _cuda_texture_resource;
    cudaResourceDesc _cuda_res_desc;
    cudaSurfaceObject_t _surface;

    cudaSurfaceObject_t _GetMappedSurface();
    void _UnmapSurface();
};

}  // namespace rt