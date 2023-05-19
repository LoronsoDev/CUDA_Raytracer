#include <iostream>
#include <raytracer/Common.h>

double hit_sphere(const Point3D& center, double radius, const Ray& r) {
    dvec3 oc = r.origin() - center;
    auto a = length2(r.direction());
    auto half_b = dot(oc, r.direction());
    auto c = length2(oc) - radius * radius;
    auto discriminant = half_b * half_b - a * c;

    if (discriminant < 0) {
        return -1.0;
    }
    else {
        return (-half_b - sqrt(discriminant)) / a;
    }
}

Color ray_color(const Ray& r) {
    auto t = hit_sphere(Point3D(0, 0, -1), 0.5, r);
    if (t > 0.0) {
        vec3 N = normalize(r.at(t) - dvec3(0, 0, -1));
        return 0.5 * (dvec3)Color(N.x + 1, N.y + 1, N.z + 1);
    }
    vec3 unit_direction = normalize(r.direction());
    t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * (dvec3)Color(1.0, 1.0, 1.0) + t * (dvec3)Color(0.5, 0.7, 1.0);
}
 
int main() {
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    // Camera

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;

    auto origin = Point3D(0, 0, 0);
    auto horizontal = dvec3(viewport_width, 0, 0);
    auto vertical = dvec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - dvec3(0, 0, focal_length);

    // Render

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            auto u = double(i) / (image_width - 1);
            auto v = double(j) / (image_height - 1);
            Ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
            Color pixel_color = ray_color(r);
            utils::write_color(std::cout, pixel_color);
        }
    }

    std::cerr << "\nDone.\n";
}