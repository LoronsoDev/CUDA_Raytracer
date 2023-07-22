#pragma once

#include <raytracer/Ray.cuh>

struct Material;

struct HitRecord {
    dvec3 p;
    dvec3 normal;
    Material* mat_ptr;
    double t;
    bool front_face;

    __device__ void set_face_normal(const Ray& r, const dvec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
    public:
	__device__ virtual bool hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const = 0;
};