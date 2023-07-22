#pragma once

#include <raytracer/Ray.cuh>
#include <raytracer/objects/Hittable.cuh>

class Sphere : public Hittable {
public:
    glm::dvec3 center;
    double radius;
    Material* material_ptr;

public:
    __device__ Sphere() {}
    __device__ Sphere(dvec3 cen, double r, Material* m) : center(cen), radius(r), material_ptr(m) {}

    __device__ bool hit(
        const Ray& r, double t_min, double t_max, HitRecord& rec) const override
    {
        glm::dvec3 oc = r.orig - center;
        auto a = length2(r.direction());
        auto half_b = dot(oc, r.direction());
        auto c = length2(oc) - radius * radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        dvec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = material_ptr;

        return true;
    }
};
