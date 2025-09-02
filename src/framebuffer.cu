#include "common.cuh"
#include "framebuffer.cuh"

namespace rt {

cudaSurfaceObject_t Framebuffer::_GetMappedSurface() {
    cudaArray_t cudaArray;
    CUDA_CHECK(cudaGraphicsMapResources(1, &_cuda_texture_resource));
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaArray, _cuda_texture_resource, 0, 0));

    _cuda_res_desc.resType = cudaResourceTypeArray;
    _cuda_res_desc.res.array.array = cudaArray;
    _cuda_res_desc.flags = 0;

    CUDA_CHECK(cudaCreateSurfaceObject(&_surface, &_cuda_res_desc));
    return _surface;
}

void Framebuffer::_UnmapSurface() {
    CUDA_CHECK(cudaDestroySurfaceObject(_surface));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &_cuda_texture_resource));
}

}  // namespace rt