#pragma once

#include "math.cuh"

namespace rt {

struct Ray {
    Vec3f origin;
    Vec3f direction;

    DEVICE_HOST Ray() : origin(0), direction(0, 0, 1) {}
    DEVICE_HOST Ray(const Vec3f& o, const Vec3f& d) : origin(o), direction(d.Normalized()) {}

    DEVICE_HOST Vec3f At(float t) const {
        return origin + direction * t;
    }
};

using Rayf = Ray;

}  // namespace rt
