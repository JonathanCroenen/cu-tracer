#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include "camera.cuh"
#include "common.cuh"
#include "hit.cuh"
#include "managed.cuh"
#include "material.cuh"
#include "math.cuh"
#include "object.cuh"

namespace rt {

using MaterialRef = unsigned int;

class Scene : public Managed {
public:
    ~Scene();
    HOST static std::unique_ptr<Scene> Create();
    HOST MaterialRef AddMaterial(const Material& material);
    HOST void AddObject(const Object& object, MaterialRef material_ref,
                        const Transformf& transform = Transformf::Identity());
    HOST void SetCamera(const Camera& camera);
    HOST void Build();

    DEVICE_HOST Rayf GetCameraRay(float u, float v) const;
    DEVICE_HOST bool Hit(const Rayf& ray, HitRecord& hit) const;

private:
    struct ObjectInstance {
        Object object;
        MaterialRef material_ref;
        Transformf transform;
    };

    Camera _camera;
    std::vector<Material> _materials;
    std::vector<ObjectInstance> _objects;

    Material* _device_materials = nullptr;
    size_t _num_materials = 0;

    ObjectInstance* _device_objects = nullptr;
    size_t _num_objects = 0;

    Scene() = default;
    HOST void _AllocateDeviceData();
    HOST void _FreeDeviceData();
};
}  // namespace rt