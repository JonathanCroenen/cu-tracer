#pragma once

#include "math.cuh"
#include "ray.cuh"

namespace rt {

struct Material;

struct HitRecord {
    Vec3f point;
    Vec3f normal;
    float t;
    bool front_face;
    Material* material;

    DEVICE_HOST
    void SetFaceNormal(const Rayf& ray, const Vec3f& outward_normal) {
        front_face = ray.direction.Dot(outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

}  // namespace rt
