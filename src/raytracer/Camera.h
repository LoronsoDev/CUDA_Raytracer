#pragma once

#include <raytracer/Common.cuh>


class Camera {
public:
    Point3D origin;
    Point3D lower_left_corner;
    dvec3 horizontal;
    dvec3 vertical;
    dvec3 u, v, w;
    double lens_radius;


public:
    __device__ __host__ Camera(Point3D lookfrom,
        Point3D lookat,
        dvec3   vup,
        double vfov, // vertical field-of-view in degrees
        double aspect_ratio,
        double aperture,
        double focus_dist)
    {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;

        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - focus_dist * w;

        lens_radius = aperture / 2.0;
    }

    __device__ Ray get_ray(double s, double t, curandState* randState) const {
        dvec3 rd = random_in_unit_disk(randState);
        dvec3 res = lens_radius * rd;
        dvec3 offset = (u * res.x) + v * res.y;

        return Ray(
            origin + offset,
            lower_left_corner + s * horizontal + t * vertical - origin - offset
        );
    }
};
