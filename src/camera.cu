#include "camera.cuh"
#include "math.cuh"

namespace rt {

DEVICE_HOST
Camera::Camera(const Vec3f& lookfrom, const Vec3f& lookat, const Vec3f& vup, float vfov,
               float aspect_ratio, float aperture, float focus_dist) {
    lens_radius = aperture / 2.0f;
    float theta = vfov * float(M_PI) / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h;
    float viewport_width = aspect_ratio * viewport_height;

    w = (lookfrom - lookat).Normalized();
    u = vup.Cross(w).Normalized();
    v = w.Cross(u);

    origin = lookfrom;
    horizontal = focus_dist * viewport_width * u;
    vertical = focus_dist * viewport_height * v;
    lower_left_corner = origin - horizontal / 2.0f - vertical / 2.0f - focus_dist * w;
}

DEVICE_HOST
Rayf Camera::GetRay(float s, float t) const {
    return Rayf(origin, lower_left_corner + s * horizontal + t * vertical - origin);
}

}  // namespace rt