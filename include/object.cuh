#pragma once

#include <cuda/std/variant>
#include "common.cuh"
#include "hit.cuh"
#include "managed.cuh"
#include "math.cuh"
#include "ray.cuh"

namespace rt {

struct Sphere {
    Vec3f center;
    float radius;

    DEVICE_HOST
    Sphere() : center(Vec3f(0, 0, 0)), radius(1.0f) {}

    DEVICE_HOST
    Sphere(const Vec3f& c, float r) : center(c), radius(r) {}

    DEVICE_HOST
    bool Hit(const Ray& ray, float t_min, float t_max, Hit& hit) const {
        Vec3f oc = ray.origin - center;
        float half_b = oc.Dot(ray.direction);
        float c = oc.LengthSquared() - radius * radius;
        float discriminant = half_b * half_b - c;

        if (discriminant < 0) return false;
        float sqrt_d = sqrtf(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float t = (-half_b - sqrt_d);
        if (t < t_min || t > t_max) {
            t = (-half_b + sqrt_d);
            if (t < t_min || t > t_max) return false;
        }

        hit.t = t;
        hit.point = ray.At(t);
        Vec3f outward_normal = (hit.point - center) / radius;
        hit.SetFaceNormal(ray, outward_normal);
        return true;
    }
};

struct Plane {
    Vec3f point;
    Vec3f normal;

    DEVICE_HOST
    Plane() : point(Vec3f(0, 0, 0)), normal(Vec3f(0, 1, 0)) {}

    DEVICE_HOST
    Plane(const Vec3f& p, const Vec3f& n) : point(p), normal(n.Normalized()) {}

    DEVICE_HOST
    Plane(const Vec3f& p0, const Vec3f& p1, const Vec3f& p2)
        : point(p0), normal((p1 - p0).Cross(p2 - p0).Normalized()) {}

    DEVICE_HOST
    bool Hit(const Ray& ray, float t_min, float t_max, Hit& hit) const {
        float denom = ray.direction.Dot(normal);
        if (denom >= 0.0f) return false;

        float t = (point - ray.origin).Dot(normal) / denom;
        if (t < t_min || t > t_max) return false;

        hit.t = t;
        hit.point = ray.At(t);
        hit.SetFaceNormal(ray, normal);
        return true;
    }
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
    bool Hit(const Ray& ray, float t_min, float t_max, Hit& hit) const {
        Vec3f edge1 = p1 - p0;
        Vec3f edge2 = p2 - p0;
        Vec3f h = ray.direction.Cross(edge2);

        float det = edge1.Dot(h);
        if (det > -epsilon && det < epsilon) return false;

        float inv_det = 1.0f / det;
        Vec3f s = ray.origin - p0;
        float u = inv_det * s.Dot(h);
        if (u < 0.0f && abs(u) > epsilon || u > 1.0f && abs(u - 1.0f) > epsilon) return false;

        Vec3f q = s.Cross(edge1);
        float v = inv_det * q.Dot(ray.direction);
        if (v < 0.0f && abs(v) > epsilon || u + v > 1.0f && abs(u + v - 1.0f) > epsilon)
            return false;

        float t = inv_det * q.Dot(edge2);
        if (t < t_min || t > t_max) return false;

        hit.t = t;
        hit.point = ray.At(t);
        hit.SetFaceNormal(ray, normal);
        return true;
    }
};

struct Object : public Managed, public cuda::std::variant<Sphere, Plane, Triangle> {
    using Base = cuda::std::variant<Sphere, Plane, Triangle>;
    using Base::Base;

    DEVICE_HOST
    bool Hit(const Ray& ray, float t_min, float t_max, Hit& hit) const {
        return cuda::std::visit(
            [&](auto&& object) -> bool { return object.Hit(ray, t_min, t_max, hit); }, *this);
    }
};

}  // namespace rt
