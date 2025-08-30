#pragma once

#include "common.cuh"

namespace rt {

class Framebuffer {
   public:
    Framebuffer() : _cuda_texture_resource(nullptr) {}
    Framebuffer(cudaGraphicsResource_t cuda_texture_resource)
        : _cuda_texture_resource(cuda_texture_resource) {}
    ~Framebuffer() {}

    void Map() {
        _GetMappedSurface();
    }

    void Unmap() {
        _UnmapSurface();
    }

    cudaSurfaceObject_t GetSurface() const {
        return _surface;
    }

   private:
    cudaGraphicsResource_t _cuda_texture_resource;
    cudaResourceDesc _cuda_res_desc;
    cudaSurfaceObject_t _surface;

    void _GetMappedSurface() {
        cudaArray_t cudaArray;
        CUDA_CHECK(cudaGraphicsMapResources(1, &_cuda_texture_resource));
        CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaArray, _cuda_texture_resource, 0, 0));

        _cuda_res_desc.resType = cudaResourceTypeArray;
        _cuda_res_desc.res.array.array = cudaArray;
        _cuda_res_desc.flags = 0;

        CUDA_CHECK(cudaCreateSurfaceObject(&_surface, &_cuda_res_desc));
    }

    void _UnmapSurface() {
        CUDA_CHECK(cudaDestroySurfaceObject(_surface));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &_cuda_texture_resource));
    }
};

}  // namespace rt