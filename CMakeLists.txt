
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wno-unused-parameter -fstrict-aliasing")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -Wall -Wextra -Wno-unused-parameter -fstrict-aliasing")

# You may need to change this if your CUDA installation is somewhere else
SET(CMAKE_C_COMPILER /opt/cuda/bin/gcc)

cmake_minimum_required (VERSION 2.6)
project (mandelgpu)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
       "Choose the type of build, options are: Debug Release"
       FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(CUDA REQUIRED)
find_package(PNG REQUIRED)

include_directories(${PROJECT_BINARY_DIR} ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} ${PNG_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})

#set(CUDA_NVCC_FLAGS_RELEASE -O3;--use_fast_math;--restrict;-arch=sm_30)
set(CUDA_NVCC_FLAGS_RELEASE -O3;--restrict;-arch=sm_30)
	
cuda_add_executable(mandelgpu image.cpp gl_renderer.cpp cuda_error.cu cuda_gl.cu mandelgpu.cpp kernel.cu)

if(USE_DOUBLE_PRECISION)
	add_definitions(-DWITH_DOUBLE_PRECISION)
endif(USE_DOUBLE_PRECISION)


target_link_libraries (mandelgpu ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${PNG_LIBRARY} ${GLEW_LIBRARIES})





