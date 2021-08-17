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

class ViewContainerBase {
public:
  virtual ~ViewContainerBase() {};
};

template<class View> 
class ViewContainer : public ViewContainerBase {
public:
  View v;
  
  ViewContainer(View &_v) : v(_v) {};
  virtual ~ViewContainer() { v.ViewClose(); }
};

struct micro_kernel_arg_t {
  struct tuple_t {
    ViewContainerBase* view;
    bool persistent;
  };
  
  std::vector<tuple_t> views;
  size_t o_sites;

  template<class T>
  void add(Lattice<T>& l, ViewMode mode, bool persistent = true) {
    size_t _o_sites = l.Grid()->oSites();
    if (views.size() == 0) {
      o_sites = _o_sites;
    } else {
      ASSERT(o_sites == _o_sites);
    }
    auto l_v = l.View(mode);
    views.push_back({ new ViewContainer<decltype(l_v)>(l_v), persistent });
  }

  void release() {
    for (auto x : views)
      delete x.view;
  }

};
