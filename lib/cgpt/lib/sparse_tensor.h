/*
    GPT - Grid Python Toolkit
    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)

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
#include <set>

class tensor_basis {
public:
  typedef std::pair<int,int> Dimension;  
  typedef std::vector<Dimension> Dimensions;
  typedef uint64_t Index;
  Dimensions dimensions;

  Index c2i(const std::vector<int>& coor) {
    Index i = 0;
    ASSERT(coor.size() == dimensions.size());
    for (size_t dim=0;dim<coor.size();dim++)
      i = i*dimensions[dim].second + coor[dim];
    return i;
  }

  std::vector<int> i2c(Index i) {
    std::vector<int> coor(dimensions.size());
    for (int dim=dimensions.size()-1;dim>=0;dim--) {
      coor[dim] = i % dimensions[dim].second;
      i /= dimensions[dim].second;
    }
    return coor;
  }

  void merge(const tensor_basis & b,
             std::vector<int>& v,
             std::vector<int>& w,
             std::set<int> exclude = std::set<int>()) {

    
    std::unordered_map<int,int> dims;
    int dim;
    
    for (dim=0;dim<dimensions.size();dim++) {
      dims[dimensions[dim].first] = dim;
    }
    
    for (dim=0;dim<b.dimensions.size();dim++) {
      int symbol = b.dimensions[dim].first;
      if (exclude.count(symbol)) {
        v.push_back(-1);
        w.push_back(-1);
      } else {
        auto f = dims.find(symbol);
        if (f == dims.end()) {
          dims[symbol] = dimensions.size();
          v.push_back(dimensions.size());
          w.push_back(-1);
          dimensions.push_back(b.dimensions[dim]);
        } else {
          v.push_back(-1);
          w.push_back(f->second);
        }
      }
    }

  }
};

class sparse_tensor {
public:

  typedef typename tensor_basis::Index Index;
  typedef std::unordered_map<Index, ComplexD> Values;
  
  Values values;

  std::shared_ptr<tensor_basis> basis;
  
  sparse_tensor(const std::shared_ptr<tensor_basis>& _basis) : basis(_basis) {
  }

  sparse_tensor operator*(const ComplexD value) {
    sparse_tensor r(basis);

    for (auto & i : values)
      r.values[i.first] = value * i.second;

    return r;
  }

  template<typename Functor>
  sparse_tensor combine(const sparse_tensor& other, Functor f) {
    // first need a joint basis
    auto r_basis = std::make_shared<tensor_basis>();

    std::vector<int> v, v_other, w, w_other;
    r_basis->merge(*basis, v, w); // v needs summing, w needs checking
    r_basis->merge(*other.basis, v_other, w_other);

    // then multiply sparse tensor
    sparse_tensor r(r_basis);
    std::vector<int> r_c = r_basis->i2c(0);
    for (auto & i : values) {
      auto c = basis->i2c(i.first);
      for (int l=0;l<c.size();l++)
        r_c[v[l]] = c[l];

      for (auto & j : other.values) {
        auto c_other = other.basis->i2c(j.first);

        int l;
        for (l=0;l<c_other.size();l++)
          if (v_other[l] != -1)
            r_c[v_other[l]] = c_other[l];
        
        for (l=0;l<c_other.size();l++)
          if (w_other[l] != -1)
            if (r_c[w_other[l]] != c_other[l])
              break;

        if (l == c_other.size()) {
          ComplexD val = f(i.second, j.second);
          if (val != 0.0) {
            auto r_i = r_basis->c2i(r_c);
            r.values[r_i] = val;
          }
        }        
      }
    }
      
    return r;
  }

  sparse_tensor operator*(const sparse_tensor& other) {
    struct {
      ComplexD operator()(ComplexD a, ComplexD b) { return a*b; };
    } op;
    return combine(other, op);
  }

  sparse_tensor operator+(const sparse_tensor& other) {
    struct {
      ComplexD operator()(ComplexD a, ComplexD b) { return a+b; };
    } op;
    return combine(other, op);
  }

  static void contract_dimension(sparse_tensor& r,
                                 std::vector<int>& t_c,
                                 std::vector<int>& r_c,
                                 const std::vector<sparse_tensor*>& other,
                                 std::vector<std::vector<int>>& tv,
                                 std::vector<std::vector<int>>& tw,
                                 std::vector<std::vector<int>>& rv,
                                 std::vector<std::vector<int>>& rw,
                                 int dim,
                                 ComplexD pv) {

    for (auto & i : other[dim]->values) {

      auto c_other = other[dim]->basis->i2c(i.first);

      int l;
      for (l=0;l<c_other.size();l++)
        if (tv[dim][l] != -1)
          t_c[tv[dim][l]] = c_other[l];

      for (l=0;l<c_other.size();l++)
        if (rv[dim][l] != -1)
          r_c[rv[dim][l]] = c_other[l];

      for (l=0;l<c_other.size();l++)
        if (tw[dim][l] != -1)
          if (t_c[tw[dim][l]] != c_other[l])
            break;

      if (l == c_other.size() && i.second != 0.0) {

        ComplexD v = pv * i.second;
        
        if (dim + 1 < other.size()) {
          contract_dimension(r, t_c, r_c, other, tv, tw, rv, rw, dim+1, v);
        } else {
          auto r_i = r.basis->c2i(r_c);
          r.values[r_i] += v;
        }
      }
    }
  }
  
  static sparse_tensor contract(const std::vector<sparse_tensor*> other,
                                std::set<int> symbols) {

    // first need a joint basis
    auto r_basis = std::make_shared<tensor_basis>();
    auto t_basis = std::make_shared<tensor_basis>();

    std::vector<std::vector<int>>
      tv(other.size()), tw(other.size()),
      rv(other.size()), rw(other.size());
    
    for (int i=0;i<other.size();i++) {
      tensor_basis obasis = *other[i]->basis;
      t_basis->merge(obasis, tv[i], tw[i]);
      r_basis->merge(obasis, rv[i], rw[i], symbols);
    }

    sparse_tensor r(r_basis);

    std::vector<int> r_c = r_basis->i2c(0);
    std::vector<int> t_c = t_basis->i2c(0);
    contract_dimension(r, t_c, r_c, other, tv, tw, rv, rw, 0, 1.0);

    return r;
  }

  void dump() {
    std::cout << "{" << std::endl;
    for (auto & i : values) {
      auto c = basis->i2c(i.first);
      std::cout << "(";
      for (size_t dim=0;dim<basis->dimensions.size();dim++) {
        if (dim>0)
          std::cout << ", ";
        std::cout << c[dim];
      }
      std::cout << ") = " << i.second << std::endl;
    }
    std::cout << "}" << std::endl;
  }
};
  /*

  

            indices = list(set(self.indices + other.indices))

            it = wick.iterator(indices)
            c = {}
            values = {}
            
            while it(c):
                new_value = tuple([ (i, c[i]) for i in indices ])
                a_value = tuple([ (i, c[i]) for i in self.indices ])
                b_value = tuple([ (i, c[i]) for i in other.indices ])

                if a_value in self.values and b_value in other.values:
                    values[new_value] = self.values[a_value] * other.values[b_value]

            return wick.tensor(
                indices,
                values
            )

        def __add__(self, other):
            indices = list(set(self.indices + other.indices))

            it = wick.iterator(indices)
            c = {}
            values = {}
            
            while it(c):
                new_value = tuple([ (i, c[i]) for i in indices ])
                a_value = tuple([ (i, c[i]) for i in self.indices ])
                b_value = tuple([ (i, c[i]) for i in other.indices ])

                terms = []
                if a_value in self.values:
                    terms.append(self.values[a_value])
                if b_value in other.values:
                    terms.append(other.values[b_value])
                if len(terms) == 1:
                    values[new_value] = terms[0]
                elif len(terms) == 2:
                    values[new_value] = terms[0] + terms[1]

            return wick.tensor(
                indices,
                values
            )

        def evaluated(self):
            return wick.tensor(
                self.indices,
                { v: g(self.values[v]) for v in self.values }
            )

        def tensor_removed(self):
            if len(self.values) == 1 and () in self.values:
                return self.values[()]
            return self

        def __setitem__(self, index_values, value):
            self.values[
                tuple([(i,j) for i, j in zip(self.indices, index_values)])
            ] = value
  */
