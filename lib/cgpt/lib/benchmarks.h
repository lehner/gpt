/*
    GPT - Grid Python Toolkit
    Copyright (C) 2020  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.


    This file provides a playground for benchmarking new C++ functions
    before they go into production.

*/

class mk_timer {
public:
  double dt_best, dt_worst, dt_total;
  size_t n;

  mk_timer() : dt_best(10000000.0), dt_worst(0.0), dt_total(0.0), n(0) {};

  void add(double dt) {
    if (dt < dt_best)
      dt_best = dt;
    if (dt > dt_worst)
      dt_worst = dt;
    dt_total += dt;
    n += 1;
  }

  void print(std::string tag, double gb) {
    std::cout << GridLogMessage << tag << ": " << gb/dt_worst << " -- " << gb/dt_best << " avg " << gb*n/dt_total << " GB/s" << std::endl;
  }
};
