#pragma once

#include <cuda/std/variant>
#include "common.cuh"
#include "hit.cuh"
#include "managed.cuh"
#include "math.cuh"
#include "ray.cuh"

namespace rt {

struct Metal {
    Vec3f albedo;
    float fuzz;

    DEVICE_HOST
    Metal(const Vec3f& albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}
    Metal(const Metal& other) = default;
    Metal(Metal&& other) = default;

    DEVICE
    bool Scatter(const Rayf& ray, const Hit& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const {
        Vec3f reflected = ray.direction.Reflect(hit.normal);
        scattered = Rayf(hit.point, reflected + fuzz * RandomUnitVector(random_state));
        if (scattered.direction.Dot(hit.normal) > 0) {
            attenuation = albedo;
            return true;
        }

        return false;
    }
};

struct Dielectric {
    Vec3f albedo;
    float refractive_index;

    DEVICE_HOST
    Dielectric(const Vec3f& albedo, float refractive_index)
        : albedo(albedo), refractive_index(refractive_index) {}
    Dielectric(const Dielectric& other) = default;
    Dielectric(Dielectric&& other) = default;

    DEVICE
    float Schlick(float cosine) const {
        float r0 = (1 - refractive_index) / (1 + refractive_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow(1 - cosine, 5);
    }

    DEVICE
    bool Refract(const Vec3f& v, const Vec3f& n, float ni_over_nt, Vec3f& refracted) const {
        Vec3f uv = v.Normalized();
        float dt = uv.Dot(n);
        float discriminant = 1 - ni_over_nt * ni_over_nt * (1 - dt * dt);

        if (discriminant > 0) {
            refracted = ni_over_nt * uv - (ni_over_nt * dt + sqrt(discriminant)) * n;
            return true;
        }

        return false;
    }

    DEVICE
    bool Scatter(const Rayf& ray, const Hit& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const {
        Vec3f reflected = ray.direction.Reflect(hit.normal);
        float ni_over_nt = hit.front_face ? (refractive_index / 1.0f) : (1.0f / refractive_index);
        Vec3f refracted;
        float reflect_prob;
        float cosine = hit.normal.Dot(-ray.direction);

        if (Refract(ray.direction, hit.normal, ni_over_nt, refracted)) {
            reflect_prob = Schlick(cosine);
        } else {
            reflect_prob = 1.0f;
        }

        if (RandomFloat(random_state) < reflect_prob) {
            scattered = Rayf(hit.point, reflected);
        } else {
            scattered = Rayf(hit.point, refracted);
        }

        attenuation = albedo;
        return true;
    }
};

struct Emissive {
    Vec3f albedo;
    float intensity;

    DEVICE_HOST
    Emissive(const Vec3f& albedo, float intensity) : albedo(albedo), intensity(intensity) {}
    Emissive(const Emissive& other) = default;
    Emissive(Emissive&& other) = default;

    DEVICE
    bool Scatter(const Rayf& ray, const Hit& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const {
        attenuation = albedo * intensity;
        return false;
    }
};

struct Material : public Managed, public cuda::std::variant<Metal, Dielectric, Emissive> {
    using Base = cuda::std::variant<Metal, Dielectric, Emissive>;
    using Base::Base;

    DEVICE
    bool Scatter(const Rayf& ray, const Hit& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const {
        return cuda::std::visit(
            [&](auto&& material) -> bool {
                return material.Scatter(ray, hit, attenuation, scattered, random_state);
            },
            *this);
    }
};

}  // namespace rt