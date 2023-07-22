#pragma once

#include <raytracer/Common.cuh>
#include <raytracer/objects/Hittable.cuh>

struct HitRecord;

class Material {
public:
    __device__ virtual bool scatter(
        const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const = 0;
};

class Lambertian : public Material
{
public:
    __device__ __host__ Lambertian(const Color& a) : albedo(a) {}

    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const override {
        auto scatter_direction = rec.normal + random_unit_vector(randState);
        if (near_zero_vector(scatter_direction))
        {
            scatter_direction = rec.normal;
        }
        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }

public:
    Color albedo;
};

//

class Metal : public Material {
public:
    __device__ __host__ Metal(const Color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState) const override {
        dvec3 reflected = reflect(normalize(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(randState));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

public:
    Color albedo;
    double fuzz;
};

//

class Dielectric : public Material {
public:
    __device__ __host__ Dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    __device__ virtual bool scatter(
        const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* randState
    ) const override {
        attenuation = Color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        dvec3 unit_direction = normalize(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        dvec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(randState))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = Ray(rec.p, direction);
        return true;
    }

public:
    double ir; // Index of Refraction

private:
    __device__ __host__ static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};