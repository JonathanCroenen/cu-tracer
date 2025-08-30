#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include "camera.cuh"
#include "common.cuh"
#include "hit.cuh"
#include "managed.cuh"
#include "material.cuh"
#include "object.cuh"

namespace rt {

using MaterialRef = unsigned int;

class Scene : public Managed {
public:
    static std::unique_ptr<Scene> Create() {
        return std::unique_ptr<Scene>(new Scene());
    }

    ~Scene() {
        _FreeDeviceData();
    }

    HOST MaterialRef AddMaterial(const Material& material) {
        _materials.push_back(material);
        return _materials.size() - 1;
    }

    HOST void AddObject(const Object& object, MaterialRef material_ref) {
        _objects.push_back(ObjectWithMaterial{object, material_ref});
    }

    HOST void SetCamera(const Camera& camera) {
        _camera = camera;
    }

    HOST void Build() {
        _AllocateDeviceData();
    }

    DEVICE_HOST
    Rayf GetCameraRay(float u, float v) const {
        return _camera.GetRay(u, v);
    }

    DEVICE_HOST
    bool Hit(const Ray& ray, Hit& hit) const {
        bool found_hit = false;
        float closest_so_far = INFINITY;

        for (size_t i = 0; i < _num_objects; i++) {
            if (_device_objects[i].object.Hit(ray, 0.001f, closest_so_far, hit)) {
                found_hit = true;
                closest_so_far = hit.t;
                hit.material = &_device_materials[_device_objects[i].material_ref];
            }
        }

        return found_hit;
    }

private:
    struct ObjectWithMaterial {
        Object object;
        MaterialRef material_ref;
    };

    Camera _camera;
    std::vector<Material> _materials;
    std::vector<ObjectWithMaterial> _objects;

    Material* _device_materials = nullptr;
    size_t _num_materials = 0;

    ObjectWithMaterial* _device_objects = nullptr;
    size_t _num_objects = 0;

    Scene() = default;

    HOST void _AllocateDeviceData() {
        _num_materials = _materials.size();
        _num_objects = _objects.size();

        CUDA_CHECK(cudaMalloc(&_device_materials, _num_materials * sizeof(Material)));
        CUDA_CHECK(cudaMalloc(&_device_objects, _num_objects * sizeof(ObjectWithMaterial)));

        CUDA_CHECK(cudaMemcpy(_device_materials, _materials.data(),
                              _num_materials * sizeof(Material), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(_device_objects, _objects.data(),
                              _num_objects * sizeof(ObjectWithMaterial), cudaMemcpyHostToDevice));
    }

    HOST void _FreeDeviceData() {
        CUDA_CHECK(cudaFree(_device_materials));
        CUDA_CHECK(cudaFree(_device_objects));

        _num_materials = 0;
        _num_objects = 0;
    }
};
}  // namespace rt