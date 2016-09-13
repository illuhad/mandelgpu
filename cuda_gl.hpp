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


#ifndef CUDA_GL
#define CUDA_GL

#include "gl_renderer.hpp"
#include <functional>

struct cudaGraphicsResource;

class cuda_gl
{
public:
  static void init_environment(int device_id = 0);
  
  cuda_gl(const gl_renderer* r);
  ~cuda_gl();
  
  void display(std::function<void (unsigned char*, std::size_t, std::size_t)> kernel_call);
  
  void rebuild_buffers();
private:
  void init();
  void release();
  
  const gl_renderer* _renderer;
  GLuint _texture;
  GLuint _buffer;
  cudaGraphicsResource* _resource;
};

#endif