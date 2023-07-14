#include <iostream>
#include <raytracer/Common.h>
#include <raytracer/objects/Hittable.h>
#include <raytracer/objects/Sphere.h>
#include <raytracer/Hittable_List.h>
#include <raytracer/Camera.h>
#include <raytracer/materials/Material.h>

#include <Raytracing.cuh>
#include <sdl/include/SDL.h>
#undef main

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

Color raytrace(const Ray& r, const Hittable& world, int depth) {    
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
 
int main() {

    // Inicializar SDL
    SDL_Init(SDL_INIT_VIDEO);

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    // Crear la ventana
    SDL_Window* window = SDL_CreateWindow("CUDA Raytracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, image_width, image_height, SDL_WINDOW_SHOWN);
    // Crear el renderizador
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

    bool quit = false;
    SDL_Event event;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255.);
            SDL_RenderClear(renderer);

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

            Color* host_pixel_colors = new Color[image_width * image_height];

            Color* device_pixel_colors;
            cudaMalloc(&device_pixel_colors, image_width * image_height * sizeof(Color));
            dim3 blockSize(16, 16);  // Tamaño del bloque: 16x16 = 256 hilos por bloque
            dim3 gridSize(image_width / blockSize.x, image_height / blockSize.y);  // Tamaño de la cuadrícula

            raytraceKernel <<< gridSize, blockSize >>> (image_width, image_height, samples_per_pixel, world, cam, device_pixel_colors);

            cudaMemcpy(host_pixel_colors, device_pixel_colors, image_width * image_height * sizeof(Color), cudaMemcpyDeviceToHost);

            for (int j = 0; j < image_height; ++j) {
                for (int i = 0; i < image_width; ++i) {
                    Color pixel_color = host_pixel_colors[j * image_width + i];
                    // Pinta el píxel en la ventana
                    SDL_SetRenderDrawColor(renderer, pixel_color.r, pixel_color.g, pixel_color.b, 255);
                    SDL_RenderDrawPoint(renderer, i, j);
                    // Presenta el renderizador en la ventana
                    SDL_RenderPresent(renderer);
                }
            }

            //for (int row = image_height - 1; row >= 0; --row) {
            //    std::cerr << "\rScanlines remaining: " << row << ' ' << std::flush;
            //    for (int col = 0; col < image_width; ++col) {
            //        Color pixel_color(0, 0, 0);
            //        for (int s = 0; s < samples_per_pixel; ++s) {
            //            auto u = (col + random_double()) / (image_width - 1);
            //            auto v = (row + random_double()) / (image_height - 1);
            //            Ray r = cam.get_ray(u, v);
            //            pixel_color += raytrace(r, world, max_depth);
            //        }
            //        //write_color(std::cout, pixel_color, samples_per_pixel);
            //        
            //        double r = pixel_color.r;
            //        double g = pixel_color.g;
            //        double b = pixel_color.b;

            //         // Divide the color by the number of samples.
            //        double scale = 1.0 / samples_per_pixel;
            //        r = sqrt(scale * r);
            //        g = sqrt(scale * g);
            //        b = sqrt(scale * b);
            //        
            //        r = 256. * clamp(r, 0.0, 0.999);
            //        g = 256. * clamp(g, 0.0, 0.999);
            //        b = 256. * clamp(b, 0.0, 0.999);

            //        unsigned int r_ = r;
            //        unsigned int g_ = g;
            //        unsigned int b_ = b;
            //        
            //        SDL_SetRenderDrawColor(renderer, (uint8) r, (uint8) g, (uint8) b, 255.);
            //        SDL_RenderDrawPoint(renderer, col, row);
            //        // Presentar el renderizador en la ventana
            //        SDL_RenderPresent(renderer);
            //    }
            //}

            std::cerr << "\nDone.\n";
            
            if (event.type == SDL_QUIT) {
                quit = true;
            }
        }
    }

    


    // Liberar recursos y cerrar SDL
    SDL_DestroyWindow(window);
    SDL_Quit();
}