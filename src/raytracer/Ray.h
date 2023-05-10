#pragma once
#include <raytracer/Common.h>

class Ray
{
public:
	dvec3 orig;
	dvec3 dir = {0.0, 0.0, 0.0};

public:
	Ray(const vec3& origin, const vec3& direction)
		:orig(origin), dir(direction)
	{
	}

	vec3 origin() const { return orig; }
	vec3 direction() const { return dir; }

	vec3 at(double t) const
	{
		return orig + t * dir;
	}
};