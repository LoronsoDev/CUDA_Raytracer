#pragma once
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

#include <raytracer/Ray.h>

using namespace glm;
using Color = fvec3;
using Point3D = dvec3;

namespace utils
{
	static void write_color(std::ostream& out, Color &pixel_color)
	{
		out << static_cast<int>(255.999 * pixel_color.r) << ' '
			<< static_cast<int>(255.999 * pixel_color.g) << ' '
			<< static_cast<int>(255.999 * pixel_color.b) << '\n';
	}
}