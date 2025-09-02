#pragma once

#include <cuda/std/variant>
#include "common.cuh"
#include "hit.cuh"
#include "managed.cuh"
#include "math.cuh"
#include "ray.cuh"

namespace rt {

struct Sphere {
    float radius;

    DEVICE_HOST
    Sphere() : radius(1.0f) {}

    DEVICE_HOST
    Sphere(float r) : radius(r) {}

    DEVICE_HOST
    bool Hit(const Rayf& ray, float t_min, float t_max, HitRecord& hit) const;
};

struct Plane {
    Vec3f normal;

    DEVICE_HOST
    Plane() : normal(Vec3f(0, 1, 0)) {}

    DEVICE_HOST
    Plane(const Vec3f& n) : normal(n.Normalized()) {}

    DEVICE_HOST
    Plane(const Vec3f& p0, const Vec3f& p1, const Vec3f& p2)
        : normal((p1 - p0).Cross(p2 - p0).Normalized()) {}

    DEVICE_HOST
    bool Hit(const Rayf& ray, float t_min, float t_max, HitRecord& hit) const;
};

struct Triangle {
    Vec3f p0;
    Vec3f p1;
    Vec3f p2;

    Vec3f normal;

    DEVICE_HOST
    Triangle()
        : p0(Vec3f(0, 0, 0)), p1(Vec3f(0, 0, 0)), p2(Vec3f(0, 0, 0)), normal(Vec3f(0, 0, 0)) {}

    DEVICE_HOST
    Triangle(const Vec3f& p0, const Vec3f& p1, const Vec3f& p2)
        : p0(p0), p1(p1), p2(p2), normal((p1 - p0).Cross(p2 - p0).Normalized()) {}

    DEVICE_HOST
    bool Hit(const Rayf& ray, float t_min, float t_max, HitRecord& hit) const;
};

struct Object : public Managed, public cuda::std::variant<Sphere, Plane, Triangle> {
    using Base = cuda::std::variant<Sphere, Plane, Triangle>;
    using Base::Base;

    DEVICE_HOST
    bool Hit(const Rayf& ray, float t_min, float t_max, HitRecord& hit) const;
};

}  // namespace rt
