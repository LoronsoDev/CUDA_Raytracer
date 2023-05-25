#pragma once

#include <raytracer/objects/Hittable.h>
#include <raytracer/Ray.h>

class Sphere : public Hittable {
public:
    Sphere() {}
    Sphere(Point3D cen, double r, shared_ptr<Material> m) : center(cen), radius(r), material_ptr(m) {};

    virtual bool hit(
        const Ray& r, double t_min, double t_max, HitRecord& rec) const override;

public:
    Point3D center;
    double radius;
    shared_ptr<Material> material_ptr;
};
