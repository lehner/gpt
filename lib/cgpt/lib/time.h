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
*/
static double cgpt_time() {
  timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec + t.tv_nsec / 1000000000.;

  //struct timeval t;
  //gettimeofday(&t,NULL);
  //return t.tv_sec + t.tv_usec / 1000000.;
}

class cgpt_timer {
 public:
  std::map<std::string,double> dt;
  std::string current_tag, title;
  double tscope;

  cgpt_timer();
  cgpt_timer(const std::string& _title);
  ~cgpt_timer();

  void operator()(const std::string& tag);
  void report();

};
