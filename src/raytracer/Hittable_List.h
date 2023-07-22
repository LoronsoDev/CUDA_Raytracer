#pragma once

#include <raytracer/objects/Hittable.cuh>
#include <raytracer/materials/Material.h>

#include <memory>
#include <vector>


using std::shared_ptr;
using std::make_shared;

class hittable_list : public Hittable {
public:
    Hittable** objects = nullptr;
    int num_objects = 0;

public:
    __device__ __host__ hittable_list(Hittable** objects_to_add, int num_objects)
    {
        objects = objects_to_add;
        this->num_objects = num_objects;
    }
	__device__    hittable_list(Hittable* object) { add(object); }
    
    __device__ void add(Hittable* object) { objects[num_objects++] = object; }

    __device__ virtual bool hit(
        const Ray& r, double t_min, double t_max, HitRecord& rec) const override;
};

__device__ bool hittable_list::hit(const Ray& r, double t_min, double t_max, HitRecord& rec) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i < num_objects; ++i) {
        bool hit = objects[i]->hit(r, t_min, closest_so_far, temp_rec);
        if (hit) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}


