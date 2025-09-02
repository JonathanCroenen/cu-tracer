#include "object.cuh"

namespace rt {

// ===== Sphere =====

DEVICE_HOST
bool Sphere::Hit(const Rayf& ray, float t_min, float t_max, HitRecord& hit) const {
    Vec3f oc = ray.origin;
    float half_b = oc.Dot(ray.direction);
    float c = oc.LengthSquared() - radius * radius;
    float discriminant = half_b * half_b - c;

    if (discriminant < 0) return false;
    float sqrt_d = sqrtf(discriminant);

    float t = (-half_b - sqrt_d);
    if (t < t_min || t > t_max) {
        t = (-half_b + sqrt_d);
        if (t < t_min || t > t_max) return false;
    }

    hit.t = t;
    hit.point = ray.At(t);
    Vec3f outward_normal = hit.point / radius;
    hit.SetFaceNormal(ray, outward_normal);
    return true;
}

// ===== Plane =====

DEVICE_HOST
bool Plane::Hit(const Rayf& ray, float t_min, float t_max, HitRecord& hit) const {
    float denom = ray.direction.Dot(normal);
    if (denom >= 0.0f) return false;

    float t = -ray.origin.Dot(normal) / denom;
    if (t < t_min || t > t_max) return false;

    hit.t = t;
    hit.point = ray.At(t);
    hit.SetFaceNormal(ray, normal);
    return true;
}

// ===== Triangle =====

DEVICE_HOST
bool Triangle::Hit(const Rayf& ray, float t_min, float t_max, HitRecord& hit) const {
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
    if (v < 0.0f && abs(v) > epsilon || u + v > 1.0f && abs(u + v - 1.0f) > epsilon) return false;

    float t = inv_det * q.Dot(edge2);
    if (t < t_min || t > t_max) return false;

    hit.t = t;
    hit.point = ray.At(t);
    hit.SetFaceNormal(ray, normal);
    return true;
}

// ===== Object =====

DEVICE_HOST
bool Object::Hit(const Rayf& ray, float t_min, float t_max, HitRecord& hit) const {
    return cuda::std::visit(
        [&](auto&& object) -> bool { return object.Hit(ray, t_min, t_max, hit); }, *this);
}

}  // namespace rt