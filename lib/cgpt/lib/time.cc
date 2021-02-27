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

// the global timer is here
cgpt_timer Timer;

// timer
cgpt_timer::cgpt_timer(bool _active) : active(_active) {
  if (_active)
    tscope = -cgpt_time();
}

cgpt_timer::~cgpt_timer() {
}

void cgpt_timer::operator()(const std::string& tag) {

  if (!active)
    return;
  
  if (current_tag.size()) {
    auto & x = dt[current_tag];
    x.first += 1;
    x.second += cgpt_time();
  }
  
  if (tag.size()) {
    dt[tag].second -= cgpt_time();
  }
  
  current_tag = tag;
}

PyObject* cgpt_timer::report() {

  PyObject* ret = PyDict_New();
  
  if (!active)
    return ret;
  
  // force stop timing
  operator()("");
  
  // total time spent
  for (auto & _dt : dt) {
    PyObject* val = PyDict_New();
    PyDict_SetItemString(ret,_dt.first.c_str(),val);

    PyObject* v = PyFloat_FromDouble(_dt.second.second);
    PyDict_SetItemString(val,"time", v); Py_XDECREF(v);

    v = PyLong_FromLong(_dt.second.first);
    PyDict_SetItemString(val,"calls", v); Py_XDECREF(v);
    
    Py_XDECREF(val);
  }

  return ret;
}

// export
EXPORT(time,{
    return PyFloat_FromDouble(cgpt_time());
  });

EXPORT(timer_begin,{
    Timer = cgpt_timer(true);
    return PyLong_FromLong(0);
  });

EXPORT(timer_end,{
    auto ret = Timer.report();
    Timer = cgpt_timer(false);
    return ret;
  });
