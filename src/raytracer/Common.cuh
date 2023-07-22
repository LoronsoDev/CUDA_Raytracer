#pragma once


#include <iostream>
#include <cstdlib>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <cuda/std/cmath>

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

// STD and lib headers
#include <cmath>
#include <limits>
#include <memory>

// Common Headers
#include <raytracer/Ray.cuh>

// Usings
using glm::dvec3;
using Color = glm::fvec3;
using Point3D = dvec3;

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

//from https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_cuda.h

template <typename T>
void check(T result, char const* const func, const char* const file,
    int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__device__ inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ static void write_color(std::ostream& out, Color &pixel_color, int samples_per_pixel)
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

__device__ inline double random_double(curandState* devState) {
	// Returns a random real in [0,1).
    auto rnd = curand_uniform_double(devState);
    return curand_uniform_double(devState);
}

__device__ inline double random_double(double min, double max, curandState* devState) {
	// Returns a random real in [min,max).
	return min + (max - min) * random_double(devState);
}

// Constants

__device__ const double infinity = std::numeric_limits<double>::infinity();

// Utility Functions

__device__ inline dvec3 random_vector(curandState* devState) {
    return dvec3(random_double(devState), random_double(devState), random_double(devState));
}

__device__ inline dvec3 random_vector(double min, double max, curandState* devState) {
    return dvec3(random_double(min, max, devState), random_double(min, max, devState), random_double(min, max, devState));
}

__device__ inline dvec3 random_in_unit_sphere(curandState* devState) {
    float r1 = random_double(devState);
    float r2 = random_double(devState);

    float phi = CURAND_2PI * r1;
    float theta = r2 * (float)CURAND_PI_DOUBLE / 2.f;
    float x = cosf(phi) * sinf(theta);
    float y = sinf(phi) * sinf(theta);
    float z = cosf(theta);
    return dvec3(x,y,z);
}

__device__ static dvec3 random_in_unit_disk(curandState* devState) {
    double x = random_double(devState) * 2.f - 1.f;
    double y = random_double(devState) * 2.f - 1.f;
    double z = 0.f;
    return normalize(dvec3(x, y, z));
}

__device__ inline bool near_zero_vector(dvec3 vector)
{
    const auto epsilon = 1e-8;
    return (fabs(vector.x) < epsilon) && (fabs(vector.y) < epsilon) && (fabs(vector.z) < epsilon);
}

__device__ inline dvec3 refract_vector(const dvec3& uv, const dvec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    dvec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    dvec3 r_out_parallel = -sqrt(fabs(1.0 - length2(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

__device__ inline dvec3 random_in_hemisphere(const dvec3& normal, curandState* devState) {
    dvec3 in_unit_sphere = random_in_unit_sphere(devState);
    if (glm::dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__device__ inline dvec3 random_unit_vector(curandState* devState) {
    return glm::normalize(random_in_unit_sphere(devState));
}

__device__ __host__ inline double degrees_to_radians(double degrees) {
	return degrees * glm::pi<double>() / 180.0;
}