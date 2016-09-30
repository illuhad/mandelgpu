/*
 * This file is part of mandelgpu, a free GPU accelerated fractal viewer,
 * Copyright (C) 2016  Aksel Alpay
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cassert>

#include "kernel.hpp"
#include "cuda_error.hpp"

#define WITHOUT_HALF //half precision is not yet supported, as it would likely (?)
                     //require a different kernel
#ifndef WITHOUT_HALF
#include <cuda_fp16.h>
#endif


const int maxiterations = 2048;
const double limit = 1000.;

__device__ 
uchar3 hsv_to_rgb(const float h,
                  const float s,
                  const float v)
{
  int h_i = h / 60;
  
  float f = h / 60.f - h_i;
  
  float p = v * (1.f - s);
  float q = v * (1.f - s * f);
  float t = v * (1.f - s * (1.f - f));
  
  float3 rgb_temp;
  
  switch(h_i)
  {
  case 0:
    rgb_temp = make_float3(v,t,p);
    break;
  case 1:
    rgb_temp = make_float3(q,v,p);
    break;
  case 2:
    rgb_temp = make_float3(p,v,t);
    break;
  case 3:
    rgb_temp = make_float3(p,q,v);
    break;
  case 4:
    rgb_temp = make_float3(t,p,v);
    break;
  case 5:
    rgb_temp = make_float3(v,p,q);
    break;
  case 6:
    rgb_temp = make_float3(v,t,p);
    break;
  }
  
  float r = rgb_temp.x * 255.f;
  float g = rgb_temp.y * 255.f;
  float b = rgb_temp.z * 255.f;
  uchar3 result = make_uchar3((unsigned char)r,
                              (unsigned char)g,
                              (unsigned char)b);
  return result;
}


__device__
uchar3 color_scheme(int num_iterations, int max_iterations)
{
  uchar3 color;
  
  if(num_iterations == max_iterations)
    color = make_uchar3(0,0,0);
  else
  {
    float h = 360.f * (0.5f * sin(num_iterations * 2.f * M_PI / (float)max_iterations + 0.2f) + 0.5f);
    float s = 0.4f * sin(num_iterations * 0.5f * M_PI / 13.f) + 0.5f;
    float v = 0.3f * sin(num_iterations * 0.5f * M_PI / 200.f) + 0.7f;
    return hsv_to_rgb(h,s,v);
  }
  
  return color;
}


template<typename Arithmetic_type, typename Arithmetic_type2>
__global__
void mandelgpu(Arithmetic_type dx, Arithmetic_type center_x, Arithmetic_type center_y,
               uchar3* pixels, int npx_x, int npx_y)
{
  int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int gid_y = threadIdx.y + blockIdx.y * blockDim.y;
  
  
  for(int px_x = gid_x; px_x < npx_x; px_x += blockDim.x * gridDim.x)
  {
    for(int px_y = gid_y; px_y < npx_y; px_y += blockDim.y * gridDim.y)
    {
      int iter_counter = 0;
      
      Arithmetic_type2 coord;
      coord.x = center_x + (px_x - npx_x / 2) * dx;
      coord.y = center_y + (px_y - npx_y / 2) * dx;
      
      Arithmetic_type2 z = coord;
      
#pragma unroll 128
      for(int i = 0; i < maxiterations; ++i)
      {         
        Arithmetic_type2 old_z = z;
        
        z = coord;
        
        Arithmetic_type a2 = 2.f * old_z.x;
        z.x += old_z.x * old_z.x;
        z.x -= old_z.y * old_z.y;
        z.y += a2 * old_z.y;
        
        Arithmetic_type norm2 = z.x * z.x;
        norm2 += z.y * z.y;

        if(norm2 > limit)
          break;
        ++iter_counter;
      }
      
      uchar3 color = color_scheme(iter_counter, maxiterations);
      pixels[px_y * npx_x + px_x] = color;
    }
  }
}


template<typename Arithmetic_type, typename Arithmetic_type2>
__global__
void juliagpu(Arithmetic_type dx, Arithmetic_type center_x, Arithmetic_type center_y,
               uchar3* pixels, int npx_x, int npx_y,
               const Arithmetic_type c0_x,
               const Arithmetic_type c0_y)
{
  int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int gid_y = threadIdx.y + blockIdx.y * blockDim.y;
  
  
  for(int px_x = gid_x; px_x < npx_x; px_x += blockDim.x * gridDim.x)
  {
    for(int px_y = gid_y; px_y < npx_y; px_y += blockDim.y * gridDim.y)
    {
      int iter_counter = 0;
      
      Arithmetic_type2 coord;
      coord.x = center_x + (px_x - npx_x / 2) * dx;
      coord.y = center_y + (px_y - npx_y / 2) * dx;
      
      Arithmetic_type2 z = coord;
      
#pragma unroll 128
      for(int i = 0; i < maxiterations; ++i)
      { 
        Arithmetic_type2 old_z = z;
        
        z.x = c0_x;
        z.y = c0_y;

        Arithmetic_type a2 = 2.f * old_z.x;
        z.x += old_z.x * old_z.x;
        z.x -= old_z.y * old_z.y;
        z.y += a2 * old_z.y;
        
        Arithmetic_type norm2 = z.x * z.x;
        norm2 += z.y * z.y;

        if(norm2 > limit)
          break;
        
        ++iter_counter;
      }

      uchar3 color = color_scheme(iter_counter, maxiterations);
      pixels[px_y * npx_x + px_x] = color;
    }
  }
  
}

template<typename Arithmetic_type, typename Arithmetic_type2>
__device__ __forceinline__
void complex_mult(Arithmetic_type2 a, Arithmetic_type2 b, Arithmetic_type2* out)
{
  out->x  = a.x * b.x;
  out->x -= a.y * b.y;
  
  out->y  = a.x * b.y;
  out->y += a.y * b.x;
}

template<typename Arithmetic_type, typename Arithmetic_type2>
__device__ __forceinline__
void complex_scale_add(Arithmetic_type2* out, Arithmetic_type2 s, Arithmetic_type2 b)
{
  out->x += s.x * b.x;
  out->x -= s.y * b.y;
  
  out->y += s.x * b.y;
  out->y += s.y * b.x;
}


template<typename Arithmetic_type, typename Arithmetic_type2, int Polynomial_degree>
__global__
void polynomial_kernel(Arithmetic_type dx, 
                   Arithmetic_type center_x, Arithmetic_type center_y,
                   uchar3* pixels, int npx_x, int npx_y,
                   Arithmetic_type2* coefficients,
                   Arithmetic_type2* coordinate_contrib_to_coefficients,
                   Arithmetic_type2 initial_value,
                   Arithmetic_type2  coordinate_contrib_to_initial_value)
{
  int gid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int gid_y = threadIdx.y + blockIdx.y * blockDim.y;
  
  Arithmetic_type2 effective_coefficients [Polynomial_degree + 1];
  Arithmetic_type2 z_powers[Polynomial_degree];
  
  for(int px_x = gid_x; px_x < npx_x; px_x += blockDim.x * gridDim.x)
  {
    for(int px_y = gid_y; px_y < npx_y; px_y += blockDim.y * gridDim.y)
    {
      int iter_counter = 0;
      
      Arithmetic_type2 coord;
      coord.x = center_x + (px_x - npx_x / 2) * dx;
      coord.y = center_y + (px_y - npx_y / 2) * dx;
      
      Arithmetic_type2 z = initial_value;
      complex_scale_add(&z, coordinate_contrib_to_initial_value, coord);
      
      // Initialise coefficients for pixel
      for(int i = 0; i < Polynomial_degree + 1; ++i)
      {
        effective_coefficients[i] = coefficients[i];
        complex_scale_add(&effective_coefficients[i], 
                          coordinate_contrib_to_coefficients[i], coord);
      }
      
      Arithmetic_type new_z;
#pragma unroll 128
      for(int i = 0; i < maxiterations; ++i)
      {
        new_z = effective_coefficients[0];
        
        // Generate powers
        z_powers[0] = z;
        complex_scale_add(&new_z, z, effective_coefficients[1]);
        
        for(int i = 1; i < Polynomial_degree; ++i)
        {
          complex_mult(z_powers[i-1], z, &z_powers[i]);
          complex_scale_add(&new_z, z_powers[i], effective_coefficients[i + 1]);
        }
        
        z = new_z;
        
        Arithmetic_type norm2 = z.x * z.x;
        norm2 += z.y * z.y;

        if(norm2 > limit)
          break;
        
        ++iter_counter;
      }

      uchar3 color = color_scheme(iter_counter, maxiterations);
      pixels[px_y * npx_x + px_x] = color;
    }
  }
}

cuda_polynomial_coefficients::cuda_polynomial_coefficients(int degree)
: _coefficients(degree + 1), _float_coefficients(degree + 1),
  _device_float_coefficients(nullptr),
  _device_double_coefficients(nullptr)
{
  cudaMalloc(&_device_float_coefficients, 2 * (degree + 1) * sizeof(float));
  cudaMalloc(&_device_double_coefficients, 2 * (degree + 1) * sizeof(double));
  
  check_cuda_error("Could not allocate device memory!");
}

cuda_polynomial_coefficients::~cuda_polynomial_coefficients()
{
  if(_device_float_coefficients)
    cudaFree(_device_float_coefficients);
  
  if(_device_double_coefficients)
    cudaFree(_device_double_coefficients);
}

void cuda_polynomial_coefficients::commit()
{
  assert(_device_float_coefficients);
  assert(_device_double_coefficients);
  
  for(std::size_t i = 0; i < _coefficients.size(); ++i)
  {
    _float_coefficients[i].real = static_cast<float>(_coefficients[i].real);
    _float_coefficients[i].imag = static_cast<float>(_coefficients[i].imag);
  }
  
  cudaMemcpy(_device_float_coefficients, _float_coefficients.data(), 
              2 * _coefficients.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(_device_double_coefficients, _coefficients.data(), 
              2 * _coefficients.size() * sizeof(double), cudaMemcpyHostToDevice);
  
  check_cuda_error("Could not transfer coefficients to device!");
}



template<typename Arithmetic_type, typename Arithmetic_type2>
performance_estimator::result run_kernel(unsigned char* pixels, 
                std::size_t width, std::size_t height,
                double size_x, 
                double center_x, double center_y,
                double c0_x, double c0_y, // for julia
                kernel_type kernel)
{
  std::size_t npx_x = width;
  std::size_t npx_y = height;

  Arithmetic_type dx = size_x / static_cast<Arithmetic_type>(npx_x);
  
  std::size_t num_bytes = npx_x * npx_y * sizeof(uchar3);

  std::size_t nthreads = 8;
  std::size_t nblocks_x = npx_x / nthreads;
  std::size_t nblocks_y = npx_y / nthreads;
  
  if(npx_x % nthreads != 0)
    ++nblocks_x;
  if(npx_y % nthreads != 0)
    ++nblocks_y;
  
  dim3 threads = dim3(nthreads, nthreads, 1);
  dim3 blocks = dim3(nblocks_x, nblocks_y, 1);

  performance_estimator perf;
  perf.start();
  
  std::size_t num_bytes_transferred = num_bytes;
  std::size_t flops = npx_x * npx_y * (maxiterations * 10);
  switch(kernel)
  {
  case MANDELBROT:
    mandelgpu<Arithmetic_type, Arithmetic_type2><<<blocks, threads>>>(
                                   static_cast<Arithmetic_type>(dx), 
                                   static_cast<Arithmetic_type>(center_x), 
                                   static_cast<Arithmetic_type>(center_y), 
                                   reinterpret_cast<uchar3*>(pixels), 
                                   static_cast<int>(npx_x), 
                                   static_cast<int>(npx_y));
    break;
  case JULIA:
    juliagpu<Arithmetic_type, Arithmetic_type2><<<blocks, threads>>>(
                                  static_cast<Arithmetic_type>(dx), 
                                  static_cast<Arithmetic_type>(center_x), 
                                  static_cast<Arithmetic_type>(center_y), 
                                  reinterpret_cast<uchar3*>(pixels), 
                                  static_cast<int>(npx_x), 
                                  static_cast<int>(npx_y),
                                  c0_x, c0_y);

    break;
  }

  check_cuda_error("Kernel execution failed!");
  cudaDeviceSynchronize();
  
  performance_estimator::result res = perf.stop(num_bytes_transferred, flops);
  
  check_cuda_error("Device synchronization failed!");
  
  return res;
}


performance_estimator::result run_kernel(unsigned char* pixels, 
                std::size_t width, std::size_t height,
                double size_x, 
                double center_x, double center_y,
                double c0_x, double c0_y, // for julia
                kernel_type kernel,
                precision p)
{
  switch(p)
  {
#ifndef WITHOUT_HALF
  case HALF:
    return run_kernel<half, half2>(pixels, 
                            width, height, 
                            size_x, center_x, center_y, 
                            c0_x, c0_y, kernel);
#endif
  case SINGLE:
    return run_kernel<float, float2>(pixels, 
                            width, height, 
                            size_x, center_x, center_y, 
                            c0_x, c0_y, kernel);
  case DOUBLE:
    return run_kernel<double, double2>(pixels, 
                            width, height, 
                            size_x, center_x, center_y, 
                            c0_x, c0_y, kernel);
  default:
    return run_kernel<double, double2>(pixels, 
                            width, height, 
                            size_x, center_x, center_y, 
                            c0_x, c0_y, kernel);
  }
  
}
