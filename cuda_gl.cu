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

#include "cuda_gl.hpp"
#include "cuda_error.hpp"
#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <cassert>

void cuda_gl::init_environment(int device_id)
{
  cudaGLSetGLDevice(device_id);
  check_cuda_error("cudaGLSetGLDevice failed!");
  glewInit();
}

cuda_gl::cuda_gl(const gl_renderer* r)
: _renderer(r)
{
  init();
}

cuda_gl::~cuda_gl()
{
  release();
}

void cuda_gl::rebuild_buffers()
{
  release();
  init();
}

void cuda_gl::release()
{
  cudaGraphicsUnregisterResource(_resource);
  glDeleteTextures(1, &_texture);
  glDeleteBuffers(1, &_buffer);
}

void cuda_gl::init()
{
  glGenTextures( 1, &_texture );
  
  glBindTexture( GL_TEXTURE_2D, _texture );

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, 
               _renderer->get_width(), _renderer->get_height(),
               0, GL_RGB, GL_UNSIGNED_BYTE, NULL );

  glBindTexture( GL_TEXTURE_2D, 0 );
  
  glGenBuffers( 1, &_buffer );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, _buffer );
  glBufferData( GL_PIXEL_UNPACK_BUFFER, 
               3 * _renderer->get_width() * _renderer->get_height(), 
               NULL, GL_STREAM_DRAW );
 
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
  

  cudaGraphicsGLRegisterBuffer(&_resource, this->_buffer, 
                               cudaGraphicsRegisterFlagsNone);
  check_cuda_error("cudaGraphicsGLRegisterBuffer failed!");
  
  assert(glGetError() == GL_NO_ERROR);
}

void cuda_gl::display(std::function<void (unsigned char*, std::size_t, std::size_t)> kernel_call)
{
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
 
  cudaGraphicsMapResources(1, &(this->_resource));
  check_cuda_error("cudaGraphicsMapResources failed!");
  
  uchar3* pixels;
  std::size_t size;
  cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&pixels), 
                                       &size, _resource);
  check_cuda_error("cudaGraphicsResourceGetMappedPointer failed!");
  
  // Call Kernel
  kernel_call(reinterpret_cast<unsigned char*>(pixels), _renderer->get_width(), _renderer->get_height());
  
  cudaGraphicsUnmapResources(1, &(this->_resource));
  check_cuda_error("cudaGraphicsUnmapResources failed!");
  
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, this->_buffer);
  glBindTexture(GL_TEXTURE_2D, this->_texture);
  
  glTexSubImage2D(GL_TEXTURE_2D, 0,
                  0, 0,
                  _renderer->get_width(), _renderer->get_height(),
                  GL_RGB, GL_UNSIGNED_BYTE, 0);
  
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  
  //glBindTexture(GL_TEXTURE_2D, _texture);
  
  
  glEnable(GL_TEXTURE_2D);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glViewport(0, 0, _renderer->get_width(), _renderer->get_height());


  glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, -1.0f);


    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, -1.0f);


    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 1.0f);


    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, 1.0f);
  glEnd();

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glDisable(GL_TEXTURE_2D);
  glGetError();
  //assert(glGetError() == GL_NO_ERROR);
}