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

#ifndef KERNEL_H
#define KERNEL_H

#include "performance.hpp"


enum kernel_type
{
  MANDELBROT,
  JULIA
};

enum precision
{
  HALF,
  SINGLE,
  DOUBLE
};

performance_estimator::result run_kernel(unsigned char* pixels, 
                std::size_t width, std::size_t height,
                double size_x, 
                double center_x, double center_y,
                double c0_x, double c0_y, // for julia
                kernel_type kernel,
                precision p);


#endif