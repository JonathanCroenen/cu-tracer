#include "common.cuh"
#include "material.cuh"

namespace rt {

// ===== Lambertian =====

DEVICE
bool Lambertian::Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                         curandState* random_state) const {
    Vec3f target = hit.point + hit.normal + RandomInUnitSphere(random_state);
    scattered = Rayf(hit.point, target - hit.point);
    attenuation = albedo;
    return true;
}

// ===== Metal =====

DEVICE
bool Metal::Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                    curandState* random_state) const {
    Vec3f reflected = ray.direction.Reflect(hit.normal);
    scattered = Rayf(hit.point, reflected + smoothness * RandomUnitVector(random_state));
    if (scattered.direction.Dot(hit.normal) > 0) {
        attenuation = albedo;
        return true;
    }

    return false;
}

// ===== Dielectric =====

DEVICE
bool Dielectric::Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                         curandState* random_state) const {
    if (hit.front_face) {
        attenuation = Vec3f(1.0f, 1.0f, 1.0f);
    } else {
        attenuation = _ApplyAbsorption(hit.t);
    }

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
    bool can_refract = _Refract(ray.direction, hit.normal, ni_over_nt, refracted);
    Vec3f reflected = ray.direction.Reflect(hit.normal);
    float reflectance = _Reflectance(abs(cosine), ior);

    if (!can_refract || RandomFloat(random_state) < reflectance) {
        Vec3f final_direction = _AddDiffusion(reflected, reflection_smoothness, random_state);
        scattered = Rayf(hit.point, final_direction);
    } else {
        Vec3f final_direction = _AddDiffusion(refracted, refraction_smoothness, random_state);
        scattered = Rayf(hit.point, final_direction);
    }

    return true;
}

DEVICE
float Dielectric::_Reflectance(float cosine, float ior) {
    // Schlick's approximation for Fresnel reflectance
    float r0 = (1.0f - ior) / (1.0f + ior);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosine, 5.0f);
}

DEVICE
bool Dielectric::_Refract(const Vec3f& incident, const Vec3f& normal, float ni_over_nt,
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
Vec3f Dielectric::_ApplyAbsorption(float distance) const {
    // Beer's Law: I(d) = I₀ * e^(-αd)
    return Vec3f(exp(-absorption_color.x * distance * absorption_factor),
                 exp(-absorption_color.y * distance * absorption_factor),
                 exp(-absorption_color.z * distance * absorption_factor));
}

DEVICE
Vec3f Dielectric::_AddDiffusion(const Vec3f& direction, float smoothness,
                                curandState* random_state) {
    if (smoothness <= 0.0f) return direction;

    Vec3f diffused = direction + RandomInUnitSphere(random_state) * smoothness;
    return diffused.Normalized();
}

// ===== Emissive =====

DEVICE
bool Emissive::Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                       curandState* random_state) const {
    attenuation = albedo * intensity;
    return false;
}

// ===== Material =====

DEVICE
bool Material::Scatter(const Rayf& ray, const HitRecord& hit, Vec3f& attenuation, Rayf& scattered,
                       curandState* random_state) const {
    return cuda::std::visit(
        [&](auto&& material) -> bool {
            return material.Scatter(ray, hit, attenuation, scattered, random_state);
        },
        *this);
}

}  // namespace rt