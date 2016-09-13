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


#include <vector>
#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <ctime>
#include <sstream>

#include "image.hpp"
#include "cuda_gl.hpp"
#include "gl_renderer.hpp"
#include "performance.hpp"
#include "kernel.hpp"

std::ostream& operator<<(std::ostream& lhs, const performance_estimator::result& rhs)
{
  lhs.precision(2);
  lhs << std::fixed;
  lhs << "Bandwidth: " << rhs.bandwidth << " GB/s. Perf: " 
      << rhs.flops << " GFlops (" << 1./rhs.time << " FPS)";
  
  return lhs;
}

class status_message
{
public:
  status_message()
  : _max_length(1)
  {}
  
  void update(const std::string& message)
  {
    if(message.size() > _max_length)
      _max_length = message.size();
    
    std::cout << '\r' << message;
    
    for(int i = 0; i < (static_cast<int>(_max_length) 
                       -static_cast<int>(message.size())); ++i)
      std::cout << " ";
    
    _current_display = message;
  }
  
  const std::string& get_current_display() const
  {
    return _current_display;
  }
private:
  std::string _current_display;
  unsigned _max_length;
};


 
class program_state_output
{
public:
  program_state_output(const std::string& initial_performance_msg,
                       const std::string& initial_mapping_msg,
                       const std::string& initial_viewport_msg)
  {
    std::cout << initial_performance_msg << std::endl;
    std::cout << initial_mapping_msg << std::endl;
    std::cout << initial_viewport_msg << std::endl;
  }
  
  void update_performance(const std::string& performance)
  {
    move_up(); move_up(); move_up();
    
    _performance_msg.update(performance);
    std::cout << std::endl;
    std::cout << _mapping_msg.get_current_display() << std::endl;
    std::cout << _viewport_msg.get_current_display() << std::endl;
  }
  
  void update_mapping(const std::string& mapping_msg)
  {
    move_up(); move_up();
    
    _mapping_msg.update(mapping_msg);
    
    std::cout << std::endl;
    std::cout << _viewport_msg.get_current_display() << std::endl;
  }
  
  void update_viewport(const std::string& viewport_msg)
  {
     move_up();
    
    _viewport_msg.update(viewport_msg);
    
    std::cout << std::endl;
  }
private:
  void move_up() const
  {
    std::cout << "\033[F";
  }
  
  status_message _performance_msg;
  status_message _mapping_msg;
  status_message _viewport_msg;
};



void build_mapping_string(kernel_type k, double c0_x, double c0_y,
                          std::string& out)
{
  if(k == MANDELBROT)
  {
    out = "Active kernel: Mandelbrot: z_(n+1) = z_n^2 + x + i*y";
  }
  else
  {
    std::stringstream sstr;
    sstr << "Active kernel: Julia: z_(n+1) = z_n^2 + " << c0_x << " + i*("<<c0_y<<")";
    out = sstr.str();
  }
}

void build_viewport_string(double center_x, double center_y, 
                           double size_x, double dx, std::string& out)
{
  std::stringstream sstr;
  sstr << "Viewport center: " << center_x << " + i*(" << center_y << "), pixel size: "
       << dx;
  out = sstr.str();
}

void usage()
{
  std::cout << "*** mandelgpu 0.1 copyright (c) 2016 Aksel Alpay *** \n";
  std::cout << "Usage:\n"
            << "  1: Select Mandelbrot kernel\n"
            << "  2: Select Julia kernel\n"
            << "  h: Reset zoom level\n"
            << "  a/d: Decrease/increase real part of\n"
            << "     constant for the Julia kernel\n"
            << "  s/w: Decrease/increase imaginary part\n"
            << "     of constant for the Julia kernel\n"
            << "  q: Quit program\n"
            << "  p: Save screenshot as 'mandelgpu.png'\n"
            << "     in current working directory\n"
            << "  Left mouse button and drag: Move view frame\n"
            << "  Right mouse button and drag vertically: Zoom in/out\n" << std::endl;
}

int main(int argc, char* argv[])
{
  usage();
  
  program_state_output state("*** Performance estimate unavailable ***",
                             "*** No mapping selected ***",
                             "*** No active viewport ***");
  
  std::size_t w = 1024;
  std::size_t h = 1024;
  
  gl_renderer::instance().init("mandelgpu", w, h, argc, argv);
  cuda_gl::init_environment();
  
  cuda_gl cgl(&gl_renderer::instance());
  
  double size_x = 1.0;
  double center_x = 0.0;
  double center_y = 0.0;
  double c0_x = 0.1;
  double c0_y = 0.5;
  double dx = size_x / 
               static_cast<double>(gl_renderer::instance().get_width());
  
  kernel_type kernel = MANDELBROT;
  precision arithmetic_precision = SINGLE;
  
  std::string mapping_str;
  build_mapping_string(kernel, c0_x, c0_y, mapping_str);
  state.update_mapping(mapping_str);
  
  std::string viewport_str;
  build_viewport_string(center_x, center_y, size_x, dx, viewport_str);
  state.update_viewport(viewport_str);
  
  auto kernel_executor = 
  [&](unsigned char* pixels, std::size_t width, std::size_t height)
  {
    performance_estimator::result r =
      run_kernel(pixels, 
                 width, height, size_x, 
                 center_x, center_y,  
                 c0_x, c0_y, 
                 kernel, 
                 arithmetic_precision);
    
    std::stringstream sstr;
    sstr << r;
    if(arithmetic_precision == SINGLE)
      sstr << " [single precision]";
    else if(arithmetic_precision == DOUBLE)
      sstr << " [double precision]";
              
    state.update_performance(sstr.str());
    
    build_viewport_string(center_x, center_y, size_x, dx, viewport_str);
    state.update_viewport(viewport_str);
  };
  
  gl_renderer::instance().on_display([&cgl, &kernel_executor]()
  {
    cgl.display(kernel_executor);
  });
  
  int old_x = 0;
  int old_y = 0;
  
  bool left_button_down = false;
  bool right_button_down = false;
  gl_renderer::instance().on_mouse(
  [&](int button, int state, int x, int y)
  {
    old_x = x;
    old_y = y;
    if(button == GLUT_LEFT_BUTTON)
    {
      if(state == GLUT_DOWN)
      {
        left_button_down = true;
      }
      else
        left_button_down = false;
    }
    else if(button == GLUT_RIGHT_BUTTON)
    {
      if(state == GLUT_DOWN)
        right_button_down = true;
      else
        right_button_down = false;
    }
  });
  
  gl_renderer::instance().on_motion(
  [&](int x, int y)
  {
    if(left_button_down)
    {
      center_x -= (x - old_x) * dx;
      center_y -= (y - old_y) * dx;
    }
    if(right_button_down)
    {
      size_x -= dx * (y - old_y);
      if(size_x < 0.0f)
        size_x = 1.e-5f;
    }
    
    dx = size_x / 
        static_cast<double>(gl_renderer::instance().get_width());
    if(dx < 3.e-8)
      arithmetic_precision = DOUBLE;
    else
      arithmetic_precision = SINGLE;
    
    old_x = x;
    old_y = y;
  });
  
  gl_renderer::instance().on_keyboard(
  [&](unsigned char c, int x, int y)
  {
    switch(c)
    {
    case '1':
      kernel = MANDELBROT;
      break;
    case '2':
      kernel = JULIA;
      break;
    case 'q':
      gl_renderer::instance().close();
      return;
    case 'a':
      if(kernel == JULIA)
        c0_x -= dx;
      break;
    case 'd':
      if(kernel == JULIA)
        c0_x += dx;
      break;
    case 'w':
      if(kernel == JULIA)
        c0_y += dx;
      break;
    case 's':
      if(kernel == JULIA)
        c0_y -= dx;
      break;
    case 'h':
      size_x = 1.0;
      arithmetic_precision = SINGLE;
      dx = size_x / 
        static_cast<double>(gl_renderer::instance().get_width());
      break;
    case 'f':
      gl_renderer::instance().toggle_fullscreen(!gl_renderer::instance().is_fullscreen());
      break;
    case 'p':
      // Write image
      gl_renderer::instance().save_png_screenshot("mandelgpu.png");
      break;
    }
    
    build_mapping_string(kernel, c0_x, c0_y, mapping_str);
    state.update_mapping(mapping_str);
  });
  
  gl_renderer::instance().on_reshape([&](int width, int height)
  {
    dx = size_x / 
            static_cast<double>(gl_renderer::instance().get_width());
    cgl.rebuild_buffers();
  });
  
  gl_renderer::instance().render_loop();

  return 0;
}

