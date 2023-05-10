#pragma once
#include <glm/glm.hpp>
#include <iostream>

using namespace glm;
using Color = vec3;

namespace utils
{
	static void write_color(std::ostream& out, Color &pixel_color)
	{
		out << static_cast<int>(255.999 * pixel_color.r) << ' '
			<< static_cast<int>(255.999 * pixel_color.g) << ' '
			<< static_cast<int>(255.999 * pixel_color.b) << '\n';
	}
}