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
#include <vector>

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

template<typename T>
struct simple_complex
{
  simple_complex(T real_part, T imag_part)
  : real(real_part), imag(imag_part)
  {}
  
  simple_complex(){}
  
  T real;
  T imag;
};

template<typename T>
std::ostream& operator<<(std::ostream& lhs, const simple_complex<T>& x)
{
  lhs << x.real;
  if(x.imag < 0.)
    lhs << x.imag;
  else
  {
    lhs << "+" << x.imag;
  }
  lhs << "*i";
  return lhs;
}

class cuda_polynomial_coefficients
{
public:
  cuda_polynomial_coefficients(int degree);
  ~cuda_polynomial_coefficients();
  
  const std::vector<simple_complex<double> >& get_coefficients() const
  { return _coefficients; }
  
  std::vector<simple_complex<double> >& get_coefficients()
  { return _coefficients; }
  
  void commit();
private:
  std::vector<simple_complex<double> > _coefficients;
  std::vector<simple_complex<float> > _float_coefficients;
  
  float* _device_float_coefficients;
  double* _device_double_coefficients;
};

performance_estimator::result run_kernel(unsigned char* pixels, 
                std::size_t width, std::size_t height,
                double size_x, 
                double center_x, double center_y,
                double c0_x, double c0_y, // for julia
                kernel_type kernel,
                precision p);


#endif