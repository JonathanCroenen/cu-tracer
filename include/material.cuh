#pragma once

#include <cuda/std/variant>
#include "common.cuh"
#include "hit.cuh"
#include "managed.cuh"
#include "math.cuh"
#include "ray.cuh"

namespace rt {

struct Lambertian {
    Vec3f albedo;

    DEVICE_HOST
    Lambertian(const Vec3f& albedo) : albedo(albedo) {}

    DEVICE
    bool Scatter(const Rayf& ray, const Hit& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const {
        Vec3f target = hit.point + hit.normal + RandomInUnitSphere(random_state);
        scattered = Rayf(hit.point, target - hit.point);
        attenuation = albedo;
        return true;
    }
};

struct Metal {
    Vec3f albedo;
    float fuzz;

    DEVICE_HOST
    Metal(const Vec3f& albedo, float fuzz) : albedo(albedo), fuzz(fuzz) {}

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
    float ior;                    // Index of refraction
    float reflection_smoothness;  // Controls diffusion of reflected rays
    float refraction_smoothness;  // Controls diffusion of refracted rays

    DEVICE_HOST
    Dielectric(float refractive_index, float refl_smoothness = 0.0f, float refr_smoothness = 0.0f)
        : ior(refractive_index), reflection_smoothness(refl_smoothness),
          refraction_smoothness(refr_smoothness) {}

    DEVICE
    static float Reflectance(float cosine, float ior) {
        // Schlick's approximation for Fresnel reflectance
        float r0 = (1.0f - ior) / (1.0f + ior);
        r0 = r0 * r0;
        return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
    }

    DEVICE
    static bool Refract(const Vec3f& incident, const Vec3f& normal, float ni_over_nt,
                        Vec3f& refracted) {
        Vec3f uv = incident.Normalized();
        float cos_theta_i = -uv.Dot(normal);
        float sin2_theta_t = ni_over_nt * ni_over_nt * (1.0f - cos_theta_i * cos_theta_i);

        if (sin2_theta_t > 1.0f) {
            return false;  // Total internal reflection
        }

        float cos_theta_t = sqrt(1.0f - sin2_theta_t);
        refracted = ni_over_nt * uv + (ni_over_nt * cos_theta_i - cos_theta_t) * normal;
        return true;
    }

    DEVICE
    static Vec3f AddDiffusion(const Vec3f& direction, float smoothness, curandState* random_state) {
        if (smoothness <= 0.0f) return direction;

        Vec3f diffused = direction + RandomInUnitSphere(random_state) * smoothness;
        return diffused.Normalized();
    }

    DEVICE
    bool Scatter(const Rayf& ray, const Hit& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const {
        attenuation = Vec3f(1.0f, 1.0f, 1.0f);

        float ni_over_nt;
        float cosine;

        if (hit.front_face) {
            // Ray is entering the material (air -> dielectric)
            ni_over_nt = 1.0f / ior;
            cosine = -ray.direction.Dot(hit.normal);
        } else {
            // Ray is exiting the material (dielectric -> air)
            ni_over_nt = ior;
            cosine = ray.direction.Dot(hit.normal);
        }

        Vec3f refracted;
        bool can_refract = Refract(ray.direction, hit.normal, ni_over_nt, refracted);
        Vec3f reflected = ray.direction.Reflect(hit.normal);
        float reflectance = Reflectance(abs(cosine), ior);

        if (!can_refract || RandomFloat(random_state) < reflectance) {
            Vec3f final_direction = AddDiffusion(reflected, reflection_smoothness, random_state);
            scattered = Rayf(hit.point, final_direction);
        } else {
            Vec3f final_direction = AddDiffusion(refracted, refraction_smoothness, random_state);
            scattered = Rayf(hit.point, final_direction);
        }

        return true;
    }
};

struct Emissive {
    Vec3f albedo;
    float intensity;

    DEVICE_HOST
    Emissive(const Vec3f& albedo, float intensity) : albedo(albedo), intensity(intensity) {}

    DEVICE
    bool Scatter(const Rayf& ray, const Hit& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const {
        attenuation = albedo * intensity;
        return false;
    }
};

struct Material : public Managed,
                  public cuda::std::variant<Lambertian, Metal, Dielectric, Emissive> {
    using Base = cuda::std::variant<Lambertian, Metal, Dielectric, Emissive>;
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