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
    bool Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const;
};

struct Metal {
    Vec3f albedo;
    float smoothness;

    DEVICE_HOST
    Metal(const Vec3f& albedo, float smoothness) : albedo(albedo), smoothness(smoothness) {}

    DEVICE
    bool Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const;
};

struct Dielectric {
    float ior;
    float reflection_smoothness;
    float refraction_smoothness;
    Vec3f absorption_color;
    float absorption_factor;

    DEVICE_HOST
    Dielectric(float refractive_index, float refl_smoothness = 0.0f, float refr_smoothness = 0.0f,
               const Vec3f& absorption_color = Vec3f(0.0f, 0.0f, 0.0f),
               const float absorption_factor = 1.0f)
        : ior(refractive_index), reflection_smoothness(refl_smoothness),
          refraction_smoothness(refr_smoothness), absorption_color(absorption_color),
          absorption_factor(absorption_factor) {}

    DEVICE
    bool Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const;

private:
    DEVICE static float _Reflectance(float cosine, float ior);
    DEVICE static bool _Refract(const Vec3f& incident, const Vec3f& normal, float ni_over_nt,
                                Vec3f& refracted);
    DEVICE Vec3f _ApplyAbsorption(float distance) const;
    DEVICE static Vec3f _AddDiffusion(const Vec3f& direction, float smoothness,
                                      curandState* random_state);
};

struct Emissive {
    Vec3f albedo;
    float intensity;

    DEVICE_HOST
    Emissive(const Vec3f& albedo, float intensity) : albedo(albedo), intensity(intensity) {}

    DEVICE
    bool Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const;
};

struct Material : public Managed,
                  public cuda::std::variant<Lambertian, Metal, Dielectric, Emissive> {
    using Base = cuda::std::variant<Lambertian, Metal, Dielectric, Emissive>;
    using Base::Base;

    DEVICE
    bool Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                 curandState* random_state) const;
};

}  // namespace rt