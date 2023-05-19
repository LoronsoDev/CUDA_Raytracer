#pragma once
#include <raytracer/Common.h>
using namespace glm;

class Ray
{
public:
	dvec3 orig;
	dvec3 dir;

public:
	Ray(const vec3& origin, const vec3& direction)
		:orig(origin), dir(direction)
	{
	}

	dvec3 origin() const { return orig; }
	dvec3 direction() const { return dir; }

	dvec3 at(double t) const
	{
		return orig + t * dir;
	}
};