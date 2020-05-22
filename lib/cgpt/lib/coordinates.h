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
template<typename coor_t>
Coordinate toCanonical(const coor_t svec) {
  if (svec.size() == 4) {
    Coordinate cvec(4);
    cvec[0] = svec[0];
    cvec[1] = svec[1];
    cvec[2] = svec[2];
    cvec[3] = svec[3];
    return cvec;
  } else if (svec.size() == 5) {
    Coordinate cvec(5);
    cvec[0] = svec[1];
    cvec[1] = svec[2];
    cvec[2] = svec[3];
    cvec[3] = svec[4];
    cvec[4] = svec[0];
    return cvec;
  } else {
    ERR("Dimension %d not supported",(int)svec.size());
  }
}

template<typename coor_t>
Coordinate fromCanonical(const coor_t cvec) {
  if (cvec.size() == 4) {
    Coordinate svec(4);
    svec[0] = cvec[0];
    svec[1] = cvec[1];
    svec[2] = cvec[2];
    svec[3] = cvec[3];
    return svec;
  } else if (cvec.size() == 5) {
    Coordinate svec(5);
    svec[0] = cvec[4];
    svec[1] = cvec[0];
    svec[2] = cvec[1];
    svec[3] = cvec[2];
    svec[4] = cvec[3];
    return svec;
  } else {
    ERR("Dimension %d not supported",(int)cvec.size());
  }
}
