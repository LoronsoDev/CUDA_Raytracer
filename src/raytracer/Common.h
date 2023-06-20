#pragma once
#include <iostream>
#include <cstdlib>

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

// STD and lib headers
#include <cmath>
#include <limits>
#include <memory>

// Common Headers
#include <raytracer/Ray.h>

// Usings
using glm::dvec3;
using Color = glm::fvec3;
using Point3D = dvec3;

using std::shared_ptr;
using std::make_shared;
using std::sqrt;


inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

static void write_color(std::ostream& out, Color &pixel_color, int samples_per_pixel)
{
    auto r = pixel_color.x;
    auto g = pixel_color.y;
    auto b = pixel_color.z;

    // Divide the color by the number of samples.
    auto scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';
}

inline double random_double() {
	// Returns a random real in [0,1).
	return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
	// Returns a random real in [min,max).
	return min + (max - min) * random_double();
}

// Constants

const double infinity = std::numeric_limits<double>::infinity();

// Utility Functions

inline dvec3 random_vector() {
    return dvec3(random_double(), random_double(), random_double());
}

inline dvec3 random_vector(double min, double max) {
    return dvec3(random_double(min, max), random_double(min, max), random_double(min, max));
}

inline dvec3 random_in_unit_sphere() {
    while (true) {
        auto p = random_vector(-1, 1);
        if (glm::length2(p) >= 1) continue;
        return p;
    }
}

static dvec3 random_in_unit_disk() {
    while (true) {
        auto p = dvec3(random_double(-1, 1), random_double(-1, 1), 0);
        if (length2(p) >= 1) continue;
        return p;
    }
}

inline bool near_zero_vector(dvec3 vector)
{
    const auto epsilon = 1e-8;
    return (fabs(vector.x) < epsilon) && (fabs(vector.y) < epsilon) && (fabs(vector.z) < epsilon);
}

inline dvec3 refract_vector(const dvec3& uv, const dvec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    dvec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    dvec3 r_out_parallel = -sqrt(fabs(1.0 - length2(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

inline dvec3 random_in_hemisphere(const dvec3& normal) {
    dvec3 in_unit_sphere = random_in_unit_sphere();
    if (glm::dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

inline dvec3 random_unit_vector() {
    return glm::normalize(random_in_unit_sphere());
}

inline double degrees_to_radians(double degrees) {
	return degrees * glm::pi<double>() / 180.0;
}