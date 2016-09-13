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

#ifndef PERFORMANCE_HPP
#define PERFORMANCE_HPP

#include <iostream>
#include <sys/time.h>
//#include <ctime>

class timer
{
  timespec start_;
  timespec stop_;
  bool is_running_;
public:
  /// Construct object

  timer()
  : is_running_(false)
  {
  }

  /// start the timer

  void start()
  {
    clock_gettime(CLOCK_MONOTONIC, &start_);
    is_running_ = true;
  }

  /// @return whether the timer is currently running

  bool is_running() const
  {
    return is_running_;
  }

  /// Stops the timer.
  /// @return The duration since the call to \c timer::start() in seconds
  /// with nanoseconds of precision

  double stop()
  {
    if (!is_running_)
      return 0.0;

    clock_gettime(CLOCK_MONOTONIC, &stop_);
    double t = stop_.tv_sec - start_.tv_sec;
    t += static_cast<double> (stop_.tv_nsec - start_.tv_nsec) * 1e-9;

    is_running_ = false;
    return t;
  }
};

class performance_estimator
{
public:
  void start()
  {
    _timer.start();
  }
  
  struct result
  {
    double bandwidth;
    double flops;
    double time;
  };
  
  result stop(std::size_t num_bytes, std::size_t num_flops)
  {
    double t = _timer.stop();
    //std::cout << num_flops << " " << t << std::endl;
    result r;
    r.bandwidth = static_cast<double>(num_bytes) * 1.e-9 / t;
    r.flops = static_cast<double>(num_flops) * 1.e-9 / t;
    r.time = t;
    
    return r;
  }
private:
  timer _timer;
};



#endif