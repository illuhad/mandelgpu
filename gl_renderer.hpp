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

#ifndef GL_RENDERER
#define GL_RENDERER

#include <functional>
#include <string>
#include <GL/glew.h>
#include <GL/glut.h>

class gl_renderer
{
public:

  static gl_renderer& instance()
  {
    static gl_renderer r;
    return r;
  }
  
  void init(const std::string& title,
              std::size_t width, std::size_t height,
              int argc, char** argv);
  
  ~gl_renderer();
  
  void close();
  
  void render_loop();
  
  
  void on_display(std::function<void ()> f);
  void on_keyboard(std::function<void (unsigned char, int, int)> f);
  void on_reshape(std::function<void(int,int)> f);
  void on_mouse(std::function<void(int,int,int,int)> f);
  void on_motion(std::function<void(int,int)> f);
  void on_idle(std::function<void()> f);
  
  std::size_t get_width() const
  { return _width; }
  
  std::size_t get_height() const
  { return _height; }
  
  void toggle_fullscreen(bool fullescreen);
  bool is_fullscreen() const;
  
  void save_png_screenshot(const std::string& name) const;
private:
  gl_renderer();
  void post_redisplay();
  
  int _width;
  int _height;
  int _window_handle;
  
  bool _is_fullscreen;
  
  friend void glut_reshape_func(int,int);
  friend void glut_display_func();
  friend void glut_keyboard_func(unsigned char c, int x, int y);
  friend void glut_mouse_func(int button, int state, int x, int y);
  friend void glut_motion_func(int x, int y);
  friend void glut_idle_func();
};

#endif
