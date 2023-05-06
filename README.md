# MandelGPU 0.11 - A free, fast, interactive fractal viewer
Copyright (c) 2016 Aksel Alpay \<Alpay@stud.uni-heidelberg.de\>

Contributions made by TheoCGaming \<theoc284@gmail.com\>

MandelGPU is a viewer for Mandelbrot and Julia fractals. It makes use of NVIDIA
CUDA for GPU acceleration (and hence unfortunately requires a NVIDIA GPU) and
OpenGL to display the images directly from the GPU without involving the CPU
too much.

# 1.) Installation

For a successful compilation, you need
- A C++11 capable compiler
- cmake       -- the build system used to compile MandelGPU
- NVIDIA CUDA -- for the GPU acceleration
- OpenGL      -- for an effcient, realtime endering of the images
- GLEW        -- for an effcient, realtime endering of the images
- GLUT        -- manages the OpenGL render window and relays mouse and keyboard input
- libpng      -- provides functionality to save screenshots as PNG file
- libpng++    -- C++ interface to libpng

If you have everything installed, create a build directory and compile MandelGPU:
```
mkdir build
cd build
cmake <Path to MandelGPU source directory>
make
```
Depending on your system, it may be necessary to change the path to the C compiler
at the start of CMakeLists.txt. This line should force cmake to use a version of gcc which
is compatible with CUDA by making it use the gcc (hopefully) provided by
your CUDA installation. CUDA is very picky about the gcc versions it works with.

# 2.) Using MandelGPU

The mandelgpu output consists of two parts: The console output providing useful
information (e.g. the current position) and the OpenGL render window which
displays the fractal.

All interaction with Mandelgpu happens by keyboard and mouse events in the OpenGL
render window.

The following keyboard or mouse events have functionality tied to them:

Fractal selection:
  1: Select Mandelbrot fractal
  ```
    The mandelbrot fractal is calculated by measuring the divergence rate of the
    complex series:
      z_(n+1) = z_n^2 + c
    where z_0 = 0 and c = x + i*y with the pixel coordinates x and y.
```
  2: Select Julia fractal
  ```
    The Julia fractal is calculated by measuring the divergence rate of the complex
    series:
      z_(n+1) = z_n^2 + c
    where z_0 = x + i*y (x and y are again the pixel coordinates) and c is a complex
    constant that can be modified by the user (see below).
```
Modifying fractals:
  a/d: Decrease/increase the real part of
       constant for Julia fractals
  s/w: Decrease/increase imaginary part
       of constant for Julia fractals

Modifying view:
  h: Reset zoom level
  f: Toggle fullscreen
  Left mouse button and drag: Move view frame
  Right mouse button and drag vertically: Zoom in/out

Other:
  q: Quit program
  p: Save screenshot as 'mandelgpu.png'
     in current working directory


If you encounter problems (or even better, have patches) please let me know.
