#include <raytracer/Hittable_List.h>
#include <raytracer/Camera.h>
#include <raytracer/materials/Material.h>

#include <cuda_runtime.h>
__global__ void raytraceKernel(int image_w, int image_h, int samples_per_pixel, hittable_list world, Camera cam, Color* device_pixel_colors);