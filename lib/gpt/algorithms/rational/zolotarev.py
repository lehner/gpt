#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2022  Mattia Bruno
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

import numpy

# zolotarev_inverse_square_root approximates 1/sqrt(x^2) with
#      (z+u_1) (z+u_2)
#  A * ------- ------- ... 
#      (z+v_1) (z+v_2)

#
# ellipj code inspired by http://www.netlib.org/cephes/
# 
def ellipj(u,m):
    EPS = 1e-16
    a = numpy.zeros((9,),dtype=numpy.float64)
    c = numpy.zeros((9,),dtype=numpy.float64)
    
    a[0] = 1.0
    b = numpy.sqrt(1.0 - m)
    c[0] = numpy.sqrt(m);
    twon = 1.0;
    i = 0;

    while ( abs(c[i]/a[i]) > EPS ):
        if( i > 7 ):
            print('Warning ellipj overflow')
            break
        ai = a[i]
        i+=1
        c[i] = ( ai - b )/2.0
        t = numpy.sqrt( ai * b )
        a[i] = ( ai + b )/2.0
        b = t
        twon *= 2.0
    
    phi = twon * a[i] * u;
    K = numpy.pi / (2.0 * a[i])
    while (i>0):
        t = c[i] * numpy.sin(phi) / a[i]
        b = phi
        phi = (numpy.arcsin(t) + phi)/2.0
        i-=1
    
    t = numpy.sin(phi)
    sn = t
    cn = numpy.cos(phi)
    dn = numpy.sqrt(1.0 - m * t * t);
    return [sn,cn,dn,K]

def zolotarev_approx_inverse_square_root(n, eps):
    a = numpy.zeros((2*n,))
    c = numpy.zeros((2*n,))

    k = numpy.sqrt(1-eps)
    _, _, _, Kk = ellipj(0, k)

    v = Kk/(2*n+1)
    for i in range(2*n):
        sn, cn, dn, _ = ellipj((i+1)*v, k)
        a[i] = (cn/sn)**2
        c[i] = sn**2
    # index go from 1 to 2*n
    c_odd = numpy.prod(c[0::2])
    c_even = numpy.prod(c[1::2])

    d = numpy.power(k, 2*n+1) * c_odd**2
    den = 1+numpy.sqrt(1-d*d)
    A = 2./den * c_odd / c_even

    delta = d**2 / den**2
    return [A, a[0::2], a[1::2], delta]

    
# f(y) = 1/sqrt(y) is approximated by A prod_{i=1}^n (y+a_{2i-1})/(y+a_{2i})
# in the range eps < y < 1; 
# the goal is to approximate g(x) = 1/sqrt(x^2) in the range ra < x < rb, 
# so we consider f(y = x^2/rb^2) / rb
# the range becomes eps*rb^2 < x^2 < rb^2, which implies eps=ra^2/rb^2
# instead all ratios (y+a_{2j-1})/(y+a_{2j}) go to (x^2 + rb^2 a_{2j-1})/(x^2 + rb^2 a_{2j})
# g(x) is approximated by (A/rb) prod_{i=1}^n (x^2 + rb^2 a_{2j-1})/(x^2 + rb^2 a_{2j})
class zolotarev_inverse_square_root:
    def __init__(self, low, high, order):
        self.ra = low
        self.rb = high
        self.n = order
        
        eps = (self.ra/self.rb)**2
        A, u, v, self.delta = zolotarev_approx_inverse_square_root(self.n, eps)
        
        self.num = u * self.rb**2
        self.poles = v * self.rb**2
        self.A = A/self.rb


    def relative_error(self, x):
        y = self.A * numpy.prod(x*x + self.num) / numpy.prod(x*x + self.poles)
        return numpy.abs(1 - y*x)

    
    def __str__(self):
        out = f"Zolotarev approx of 1/sqrt(x^2) with {self.n} poles in the range [{self.ra},{self.rb}]\n"
        out += f"   relative error delta = {self.delta}"
        return out
    