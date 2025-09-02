#pragma once

#include "math.cuh"
#include "ray.cuh"

namespace rt {

struct Camera {
    Vec3f origin;
    Vec3f lower_left_corner;
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f u, v, w;
    float lens_radius;

    DEVICE_HOST
    Camera()
        : origin(Vec3f(0, 0, 0)), lower_left_corner(Vec3f(0, 0, 0)), horizontal(Vec3f(0, 0, 0)),
          vertical(Vec3f(0, 0, 0)), u(Vec3f(0, 0, 0)), v(Vec3f(0, 0, 0)), w(Vec3f(0, 0, 0)),
          lens_radius(0.0f) {}

    DEVICE_HOST
    Camera(const Vec3f& lookfrom, const Vec3f& lookat, const Vec3f& vup, float vfov,
           float aspect_ratio, float aperture = 0.0f, float focus_dist = 1.0f);

    DEVICE_HOST
    Rayf GetRay(float s, float t) const;
};

}  // namespace rt
