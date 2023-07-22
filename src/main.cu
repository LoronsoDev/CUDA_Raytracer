#define GLM_FORCE_CUDA

#include <iostream>

#include <raytracer/Common.cuh>
#include <raytracer/objects/Hittable.cuh>
#include <raytracer/objects/Sphere.cuh>
#include <raytracer/Hittable_List.h>
#include <raytracer/Camera.h>
#include <raytracer/materials/Material.h>

#include <sdl/include/SDL.h>
#undef main

__device__ Color raytrace(const Ray r, const hittable_list** d_world, int max_depth, curandState* rand_state) {
	Color attenuation_acc = Color(1.0f, 1.0f, 1.0f);
	Ray ray = r;

	for (int i = 0; i < max_depth; i++)
	{
		HitRecord rec;
		if ((*d_world)->hit(ray, 0.001, FLT_MAX, rec)) {
			Ray scattered;
			Color attenuation;
			if (rec.mat_ptr->scatter(ray, rec, attenuation, scattered, rand_state))
			{
				ray = scattered;
				attenuation_acc *= attenuation;
			}
			else
			{
				return Color(0,0,0);
			}
		}
		else
		{
			dvec3 unit_direction = normalize(r.direction());
			auto t = 0.5 * (unit_direction.y + 1.0);
			auto color_res = (1.0 - t) * (dvec3)Color(1.0, 1.0, 1.0) + t * (dvec3)Color(0.5, 0.7, 1.0);
			return color_res * (dvec3)attenuation_acc;
		}
	}

	return Color(0,0,0);
}

constexpr int world_size = 4; //num of objects in the world
//Need to be allocated before being called (can't alloc memory from kernels)
__global__ void create_world(hittable_list** d_world, Hittable** instance_holder, int size)
{
	//We only want this to happen once
	if (threadIdx.x != 0 && blockIdx.x != 0) return;

	auto material_ground = new Lambertian(Color(0.8, 0.8, 0.0));
	auto material_center = new Lambertian(Color(0.1, 0.2, 0.5));
	auto material_left = new Dielectric(1.5);
	auto material_right = new Metal(Color(0.8, 0.6, 0.2), 0.8);

	auto radius = 0.7;

	int i = 0;
	instance_holder[i++] = new Sphere(dvec3(0.0, -100.5, -1.0), 100.0, material_ground);
	instance_holder[i++] = new Sphere(dvec3(0.0, 0.0, -1.0), 0.5, material_center);
	instance_holder[i++] = new Sphere(dvec3(-radius, 0.0, -1.0), radius, material_left);
	instance_holder[i] = new Sphere(dvec3(radius, 0.0, -1.0), radius, material_right);

	*d_world = new hittable_list(instance_holder, size);
}

const auto aspect_ratio = 16.0 / 9.0;
const int image_width = 400;
const int image_height = static_cast<int>(image_width / aspect_ratio);
const int samples_per_pixel = 1;
const int max_depth = 50;

__global__ void raytraceKernel(int image_w, int image_h, int samples_per_pixel, hittable_list** d_world, const Camera* d_cam, Color* device_pixel_colors, curandState* rand_state)
{
	int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;          //i
	int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;          //j

	if (pixel_x > image_w || pixel_y > image_h)
	{
		return;
	}

	int uid = pixel_y * image_w + pixel_x; //Unique identifier for this pixel

	curandState pixel_state = rand_state[uid];

	Color pixel_color(0.1, 0.1, 1);

	double rand1 = random_double(&pixel_state);
	double rand2 = random_double(&pixel_state);

	double u = (pixel_x + rand1) / (double)(image_w - 1);
	double v = (pixel_y + rand2) / (double)(image_h - 1);
	Ray ray = d_cam->get_ray(u, v, &pixel_state);
	pixel_color = raytrace(ray, d_world, max_depth, &pixel_state);

	/* for (int s = 0; s < samples_per_pixel; ++s) {
	 }*/

	 //double r = pixel_color.r;
	 //double g = pixel_color.g;
	 //double b = pixel_color.b;
	 //
	 //// Divide the color by the number of samples.
	 //double scale = 1.0 / samples_per_pixel;
	 //r = sqrt(scale * r);
	 //g = sqrt(scale * g);
	 //b = sqrt(scale * b);
	 //
	 //r = 256. * clamp(r, 0.0, 0.999);
	 //g = 256. * clamp(g, 0.0, 0.999);
	 //b = 256. * clamp(b, 0.0, 0.999);

	 //pixel_color.r = r;
	 //pixel_color.g = g;
	 //pixel_color.b = b;

	device_pixel_colors[image_w * (image_h - pixel_y) + pixel_x] = pixel_color;
}

__global__ void initCurand(curandState* state, unsigned long seed, unsigned w, unsigned h) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > w || y > h)
		return;

	int index = y * w + x;

	curand_init(seed, index, 0, &state[index]);
}

int main() {
	// Inicializar SDL
	SDL_Init(SDL_INIT_VIDEO);

	// Crear la ventana
	SDL_Window* window = SDL_CreateWindow("CUDA Raytracer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, image_width, image_height, SDL_WINDOW_SHOWN);
	// Crear el renderizador
	SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, 0);

	bool quit = false;
	SDL_Event event;
	while (!quit) {
		SDL_PollEvent(&event);

		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255.);
		SDL_RenderClear(renderer);

		// World

		Hittable** d_instance_holder;
		hittable_list** d_world;
		checkCudaErrors(cudaMalloc((void**)&d_instance_holder, world_size * sizeof(Hittable)));
		checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable_list)));

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, 0);

		//auto threads_per_block = sqrt(512);
		dim3 blockSize(16, 16);              // Tamaño del bloque
		dim3 gridSize(
			ceil((image_width + blockSize.x - 1) / (float)blockSize.x),
			ceil((image_height + blockSize.y - 1) / (float)blockSize.y));                  // Tamaño de la cuadrícula

		create_world << < 1, 1 >> > (d_world, d_instance_holder, world_size); //We only want to call this once
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		assert(cudaGetLastError() == 0); //assert there's no errors

		// Camera
		Point3D lookfrom(3, 3, 2);
		Point3D lookat(0, 0, -1);
		dvec3 vup(0, 1.0, 0);
		double dist_to_focus = length((lookfrom - lookat));
		double aperture = 0.0; //blur

		Camera* cam = new Camera(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);
		Camera* d_cam;

		cudaMalloc((void**)&d_cam, sizeof(Camera));
		checkCudaErrors(cudaMemcpy(d_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice));

		//// Render

		std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

		Color* h_pixel_colors = new Color[image_width * image_height];
		std::fill(h_pixel_colors, h_pixel_colors + image_width * image_height, Color(1, 0, 0.5));

		Color* d_pixel_colors;

		checkCudaErrors(cudaMalloc((void**)&d_pixel_colors, image_width * image_height * sizeof(Color)));
		checkCudaErrors(cudaMemcpy(d_pixel_colors, h_pixel_colors, image_width * image_height * sizeof(Color), cudaMemcpyHostToDevice)); //Not needed but useful for debugging.

		curandState* devState;
		checkCudaErrors(cudaMalloc((void**)&devState, 8192 * 16 * sizeof(curandState)));
		initCurand << < gridSize, blockSize >> > (devState, 7355608, image_width, image_height);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());

		raytraceKernel << < gridSize, blockSize >> > (image_width, image_height, samples_per_pixel, d_world, d_cam, d_pixel_colors, devState);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		assert(cudaGetLastError() == 0); //assert there's no errors
		printf("Raytrace finished - showing render");

		checkCudaErrors(cudaMemcpy(h_pixel_colors, d_pixel_colors, image_width * image_height * sizeof(Color), cudaMemcpyDeviceToHost));

		for (int j = image_height - 1; j >= 0; --j) {
			for (int i = 0; i < image_width; ++i) {
				Color pixel_color = h_pixel_colors[j * image_width + i];
				// Pinta el píxel en la ventana
				SDL_SetRenderDrawColor(renderer, pixel_color.r * 255, pixel_color.g * 255, pixel_color.b * 255, 255);
				SDL_RenderDrawPoint(renderer, i, j);
				// Presenta el renderizador en la ventana
				SDL_RenderPresent(renderer);
			}
		}

		std::cerr << "\nDone.\n";

		if (event.type == SDL_QUIT) {
			quit = true;
		}
	}

	// Liberar recursos y cerrar SDL
	SDL_DestroyWindow(window);
	SDL_Quit();
}