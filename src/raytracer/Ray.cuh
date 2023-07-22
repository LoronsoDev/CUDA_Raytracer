#pragma once
#include <raytracer/Common.cuh>

class Ray
{
public:
	glm::dvec3 orig;
	glm::dvec3 dir;

public:
	__device__ Ray() = default;

	__device__ Ray(const glm::vec3 origin, const glm::vec3 direction)
		:orig(origin), dir(direction)
	{
	}

	__device__ glm::dvec3 origin() const { return orig; }
	__device__ glm::dvec3 direction() const { return dir; }

	__device__ glm::dvec3 at(double t) const
	{
		return orig + t * dir;
	}
};