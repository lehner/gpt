# Author: Christopher Kelly 2026
import gpt as g
from gpt.algorithms.rational.remez import *
import gpt.algorithms.inverter as ginv
import math

#Test x^{1/2} Remez approximation
approx = Remez(1,2, 0.1,1,10, precision=200)

x = np.linspace(approx.lo,approx.hi,500)
expect = approx.func(x)
got = approx.approx(x)
g.message(f"Test rational approximation of x^(1/2): {np.linalg.norm(expect-got)} (expect 0)")

#Test partial-fraction expansion
pfe_approx = RemezPFE(approx, precision=100)
got = pfe_approx.approx(x)
g.message(f"Test PFE of rational approximation of x^(1/2): {np.linalg.norm(expect-got)} (expect 0)")

#Test rational_function wrapper
rat = pfe_approx.rationalFunction()
assert isinstance(rat, rational_function)
got = np.array([ rat(xx) for xx in x ])
g.message(f"Test PFE of rational approximation of x^(1/2) via rational_function: {np.linalg.norm(expect-got)} (expect 0)")

#Test PFE of inverse
expect = x**(-1/2)
got = pfe_approx.approx(x, inv=True)
n2 = np.linalg.norm(expect-got)
g.message(f"Test PFE of rational approximation of x^(-1/2): {n2} (expect 0)")

#Test rational_function wrapper
rat_inv = pfe_approx.rationalFunction(inv=True)
assert isinstance(rat_inv, rational_function)
got = np.array([ rat_inv(xx) for xx in x ])
g.message(f"Test PFE of rational approximation of x^(-1/2) via rational_function: {np.linalg.norm(expect-got)} (expect 0)")

rat_inv = rat.inv()
assert isinstance(rat_inv, rational_function)
got = np.array([ rat_inv(xx) for xx in x ])
eps = np.linalg.norm(expect-got)
g.message(f"Test PFE of rational approximation of x^(1/2) via rational_function.inv: {eps} (expect 0)")
assert eps < 1e-13

#Demonstrate rational function applied to lattice-matrix via rational_function
grid = g.grid([4, 4, 4, 4], g.double)

#Use a diagonal 4x4 matrix so we can easily test the inverse
Asite = g.tensor( np.array([ [2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2] ]), g.ot_matrix_singlet(4) )
A = g.lattice(grid, g.ot_matrix_singlet(4) )
A[:] = Asite

def applyA(Av, v):
    Av @= A *v

inv_mcg = ginv.multi_shift_cg(eps=1e-8, maxiter=10000, shifts=[ -p for p in pfe_approx.pfe_poles ])
rat = pfe_approx.rationalFunction(inverter=inv_mcg)
ratA = rat(applyA)

v = g.lattice(grid, g.ot_vector_singlet(4) )
vsite = g.tensor(np.array([1,0,0,0]), g.ot_vector_singlet(4) )  #just pull out the first column of A^{1/2}
v[:] = vsite

out = g.copy(v)
ratA(out, v)

d=out[(0,0,0,0)][0] - np.complex128(math.sqrt(2))
g.message(f"Test A^(1/2) for A=diag(2,2,2,2): {d} (expect 0)")
assert abs(d) < 1e-14
