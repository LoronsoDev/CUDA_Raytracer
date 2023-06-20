#include <iostream>
#include <raytracer/Common.h>
#include <raytracer/objects/Hittable.h>
#include <raytracer/objects/Sphere.h>
#include <raytracer/Hittable_List.h>
#include <raytracer/Camera.h>
#include <raytracer/materials/Material.h>

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

Color ray_color(const Ray& r, const Hittable& world, int depth) {    
    HitRecord rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return Color(0, 0, 0);

    if (world.hit(r, 0.001, infinity, rec)) {
        Ray scattered;
        Color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth - 1);
        return Color(0, 0, 0);
    }
    vec3 unit_direction = normalize(r.direction());
    auto t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * (dvec3)Color(1.0, 1.0, 1.0) + t * (dvec3)Color(0.5, 0.7, 1.0);
}
 
int main() {
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    // World
    hittable_list world;
    auto radius = cos(pi<double>() / 4);

    auto material_ground = make_shared<Lambertian>(Color(0.8, 0.8, 0.0));
    auto material_center = make_shared<Lambertian>(Color(0.1, 0.2, 0.5));
    auto material_left = make_shared<Dielectric>(1.5);
    auto material_right = make_shared<Metal>(Color(0.8, 0.6, 0.2), 0.8);

    world.add(make_shared<Sphere>(Point3D(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(make_shared<Sphere>(Point3D(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(make_shared<Sphere>(Point3D(-radius, 0.0, -1.0), radius, material_left));
    world.add(make_shared<Sphere>(Point3D(radius, 0.0, -1.0), radius, material_right));

    // Camera
    Point3D lookfrom(3, 3, 2);
    Point3D lookat(0, 0, -1);
    dvec3 vup(0, 1.0, 0);
    double dist_to_focus = length((lookfrom - lookat));
    double aperture = 2.0;

    Camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

    // Render

    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    
    for (int j = image_height - 1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            Color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width - 1);
                auto v = (j + random_double()) / (image_height - 1);
                Ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";
}