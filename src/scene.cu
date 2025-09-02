#include "scene.cuh"

namespace rt {

std::unique_ptr<Scene> Scene::Create() {
    return std::unique_ptr<Scene>(new Scene());
}

Scene::~Scene() {
    _FreeDeviceData();
}

HOST MaterialRef Scene::AddMaterial(const Material& material) {
    _materials.push_back(material);
    return _materials.size() - 1;
}

HOST void Scene::AddObject(const Object& object, MaterialRef material_ref,
                           const Transformf& transform) {
    _objects.push_back(ObjectInstance{object, material_ref, transform});
}

HOST void Scene::SetCamera(const Camera& camera) {
    _camera = camera;
}

HOST void Scene::Build() {
    _AllocateDeviceData();
}

DEVICE_HOST
Rayf Scene::GetCameraRay(float u, float v) const {
    return _camera.GetRay(u, v);
}

DEVICE_HOST
bool Scene::Hit(const Rayf& ray, HitRecord& hit) const {
    bool found_hit = false;
    float closest_so_far = INFINITY;

    for (size_t i = 0; i < _num_objects; i++) {
        const auto& obj = _device_objects[i];
        Rayf local_ray = obj.transform.TransformRay(ray);

        if (obj.object.Hit(local_ray, 0.001f, closest_so_far, hit)) {
            found_hit = true;
            closest_so_far = hit.t;
            hit.material = &_device_materials[obj.material_ref];

            // Transform the hit point and normal back to world space
            hit.point = obj.transform.TransformPoint(hit.point);
            hit.normal = obj.transform.TransformNormal(hit.normal);
        }
    }

    return found_hit;
}

HOST void Scene::_AllocateDeviceData() {
    _num_materials = _materials.size();
    _num_objects = _objects.size();

    CUDA_CHECK(cudaMalloc(&_device_materials, _num_materials * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&_device_objects, _num_objects * sizeof(ObjectInstance)));

    CUDA_CHECK(cudaMemcpy(_device_materials, _materials.data(), _num_materials * sizeof(Material),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_device_objects, _objects.data(), _num_objects * sizeof(ObjectInstance),
                          cudaMemcpyHostToDevice));
}

HOST void Scene::_FreeDeviceData() {
    CUDA_CHECK(cudaFree(_device_materials));
    CUDA_CHECK(cudaFree(_device_objects));

    _num_materials = 0;
    _num_objects = 0;
}

}  // namespace rt