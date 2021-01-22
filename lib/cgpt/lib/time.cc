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
#include "lib.h"

// timer
cgpt_timer::cgpt_timer() {
  tscope = -cgpt_time();
}

cgpt_timer::cgpt_timer(const std::string& _title) : title(_title) {
  tscope = -cgpt_time();
}

cgpt_timer::~cgpt_timer() {
  //tscope += cgpt_time();
  //std::cout << GridLogMessage << std::setw(50) << (" Time in scope " + title) << std::setw(24) << std::right << tscope << " s" << std::endl;
}

void cgpt_timer::operator()(const std::string& tag) {
  
  if (current_tag.size()) {
    dt[current_tag] += cgpt_time();
  }
  
  if (tag.size()) {
    dt[tag] -= cgpt_time();
  }
  
  current_tag = tag;
}

void cgpt_timer::report() {
  
  // force stop timing
  operator()("");
  
  std::cout << GridLogMessage << "================================================================================" << std::endl;
  std::cout << GridLogMessage << "  Timing report " << title << std::endl;
  std::cout << GridLogMessage << "================================================================================" << std::endl;
  
  // total time spent
  double tot = 0.0;
  for (auto & _dt : dt)
    tot += _dt.second;
  
  // sort
  std::vector<std::string> tags;
  for (auto & _dt : dt)
    tags.push_back(_dt.first);
  std::sort(tags.begin(), tags.end(), [=](std::string a, std::string b) {return dt[a] > dt[b]; });
  
  for (auto & tag : tags) {
    double time = dt[tag];
    std::cout << GridLogMessage <<  " " <<
      std::setw(40) << tag << 
      std::setw(15) << std::right << time/ tot * 100 << " % " <<
      std::setw(15) << std::right << time << " s"<< std::endl;
  }
  
  std::cout << GridLogMessage << "================================================================================" << std::endl;
  std::cout << GridLogMessage << std::setw(42) << " Total time " << std::setw(32) << std::right << tot << " s" << std::endl;
}

// export
EXPORT(time,{
    return PyFloat_FromDouble(cgpt_time());
  });
