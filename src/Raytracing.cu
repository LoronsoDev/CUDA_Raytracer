#include <raytracer/objects/Sphere.h>
#include <Raytracing.cuh>

const auto aspect_ratio = 16.0 / 9.0;
const int image_width = 400;
const int image_height = static_cast<int>(image_width / aspect_ratio);
const int samples_per_pixel = 100;
const int max_depth = 50;


__device__ bool hit(const hittable_list world, const Ray& r, double t_min, double t_max, HitRecord& rec)
{
    HitRecord temp_rec;
    bool hit_anything = false;
    auto closest_so_far = t_max;

    for (const auto& object : objects) {
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
}

__device__ Color raytrace(const Ray& r, const Hittable& world, int depth) {
    HitRecord rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return Color(0, 0, 0);

    if (world.hit(r, 0.001, infinity, rec)) {
        Ray scattered;
        Color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * raytrace(scattered, world, depth - 1);
        return Color(0, 0, 0);
    }
    vec3 unit_direction = normalize(r.direction());
    auto t = 0.5 * (unit_direction.y + 1.0);
    return (1.0 - t) * (dvec3)Color(1.0, 1.0, 1.0) + t * (dvec3)Color(0.5, 0.7, 1.0);
}

__global__ void raytraceKernel(int image_w, int image_h, int samples_per_pixel, hittable_list world, Camera cam, Color* device_pixel_colors)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;                   //i
    int row = image_h - 1 - (blockIdx.y * blockDim.y + threadIdx.y);   //j

    if (col >= image_w || row < 0)
        return;

    Color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = (col + random_double()) / (image_w - 1);
        auto v = (row + random_double()) / (image_h - 1);
        Ray r = cam.get_ray(u, v);
        pixel_color += raytrace(r, world, max_depth);
    }

    double r = pixel_color.r;
    double g = pixel_color.g;
    double b = pixel_color.b;

    // Divide the color by the number of samples.
    double scale = 1.0 / samples_per_pixel;
    r = sqrt(scale * r);
    g = sqrt(scale * g);
    b = sqrt(scale * b);

    r = 256. * clamp(r, 0.0, 0.999);
    g = 256. * clamp(g, 0.0, 0.999);
    b = 256. * clamp(b, 0.0, 0.999);

    pixel_color.r = r;
    pixel_color.g = g;
    pixel_color.b = b;

    device_pixel_colors[row * image_width + col] = pixel_color;
}