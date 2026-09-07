#!/usr/bin/env python3
#
# Authors: Christoph Lehner
# Acknowledgements: Qwen 3.8 27b in Pi Coding Agent
#
# Desc.: Higher-order (2nd and 3rd) derivatives in the reverse-accumulation
#        AD framework via nested nodes.
#
# A k-th order derivative is obtained with k nested rad.node() wraps
# n_k = node(node(...node(x)...)).  After evaluating the action graph
# S(n_k)(), the leaf's .gradient is a (lazy) compute graph for dS/dx.
# Contracting that graph with a constant direction node (created with
# with_gradient=False) and evaluating the result runs the next reverse
# pass, depositing the next derivative into the next-inner leaf's
# .gradient, e.g. for n3 = node(node(node(x))):
#
#   S(n3)()                          -> n3.gradient           graph for dS/dx
#   inner_product(na, n3.gradient)() -> n3.value.gradient     graph for d2S/dx2 * a
#   inner_product(nb, n3.value.gradient)()
#                                     -> n3.value.value.gradient = d3S/dx3 * a * b
#
# Conventions:
#  - the contraction must be linear in the gradient argument.  For lattice
#    nodes, g.inner_product(direction, gradient) puts the gradient in the
#    linear (second) slot; for scalar nodes use g.adj(direction_node) *
#    gradient (node __mul__ is adjoint-linear in its first argument).
#  - for SU(N) gauge fields, contract with g.group.inner_product as in
#    applications/hmc/hessian.py.
#
import gpt as g
from gpt.ad.reverse.util import resolve as resolve_value

rng = g.random("test")
rad = g.ad.reverse


def assert_close(val, ref, tol, msg):
    val = resolve_value(val)
    err = abs(val - ref) / (abs(val) + abs(ref) + 1e-30)
    g.message(f"{msg}: {err} {val} {ref}")
    assert err < tol, msg


def assert_field_close(f, ref, tol, msg):
    f = resolve_value(f)
    err = g.norm2(f - ref) / g.norm2(ref)
    g.message(f"{msg}: {err}")
    assert err < tol, msg


#####################################
# stage 0: complex scalar
#####################################
g.message("Testing higher-order derivatives on a complex scalar")

# S(x) = x^4 + 3 x^2, exact derivatives:
#   dS   = 4 x^3 + 6 x
#   d2S* = (12 x^2 + 6) a
#   d3S* = 24 x a b
#
# x, a, b are real: node __mul__ is adjoint-linear in its arguments, so a
# constant captured in the gradient graph would be conjugated for complex
# variables.  (Gauge-field directions are skew-Hermitian, i.e., effectively
# real, and are not affected.)
for x0 in [1.333, -0.7, 0.3]:
    a = 2.0
    b = 0.5

    n = rad.node(x0)
    S = n**4 + 3.0 * n**2
    S()
    assert_close(n.gradient, 4 * x0**3 + 6 * x0, 1e-14, "scalar dS/dx")

    n2 = rad.node(rad.node(x0))
    S2 = n2**4 + 3.0 * n2**2
    S2()
    na = rad.node(a, with_gradient=False)
    c2 = g.adj(na) * n2.gradient
    c2()
    assert_close(
        n2.value.gradient, (12 * x0**2 + 6) * a, 1e-13, "scalar d2S/dx2 * a"
    )

    n3 = rad.node(rad.node(rad.node(x0)))
    S3 = n3**4 + 3.0 * n3**2
    S3()
    nb = rad.node(b, with_gradient=False)
    c3 = g.adj(na) * n3.gradient
    c3()
    c3b = g.adj(nb) * n3.value.gradient
    c3b()
    assert_close(
        n3.value.value.gradient, 24 * x0 * a * b, 1e-12, "scalar d3S/dx3 * a * b"
    )


#####################################
# stage 1: complex singlet lattice
#####################################
g.message("Testing higher-order derivatives on a complex lattice")

grid = g.grid([4, 4, 4, 4], g.double)
rng_l = g.random("test_lattice")

s0 = g.complex(grid)
rng_l.cnormal(s0)
a0 = rng_l.cnormal(g.complex(grid))
b0 = rng_l.cnormal(g.complex(grid))
na = rad.node(a0, with_gradient=False)
nb = rad.node(b0, with_gradient=False)

# --- exact case: S = sum |s|^4, with C = s * adj(s) (real) ---
# exact derivatives (dS = Re sum adj(G) ds):
#   G(s)   = 4 C s
#   H(a)   = 4 C a + 4 (adj(s) a + s adj(a)) s
#   T(a, b) = 4 (2 adj(s) a + adj(a) s + s adj(a)) b
#            + 4 (s a + a s) adj(b)
#  (d/ds acts on adj(s) as d(adj(s))/ds = adj(d s), so the third derivative
#  is dH/ds = dH/ds| + adj(dH/dadj(s)|) with b substituted)
C0 = s0 * g.adj(s0)

g.message("quartic norm action: exact derivatives")


def quartic_norm(n):
    C = n * g.adj(n)
    return g.sum(C * C)


n = rad.node(s0)
quartic_norm(n)()
assert_field_close(n.gradient, 4 * C0 * s0, 1e-13, "quartic norm dS/ds")

n2 = rad.node(rad.node(s0))
quartic_norm(n2)()
c2 = g.inner_product(na, n2.gradient)
c2()
ref_h = 4 * C0 * a0 + 4 * (g.adj(s0) * a0 + s0 * g.adj(a0)) * s0
assert_field_close(n2.value.gradient, ref_h, 1e-12, "quartic norm HVP")

n3 = rad.node(rad.node(rad.node(s0)))
quartic_norm(n3)()
c3 = g.inner_product(na, n3.gradient)
c3()
c3b = g.inner_product(nb, n3.value.gradient)
c3b()
ref_t = (
    4 * (2 * g.adj(s0) * a0 + g.adj(a0) * s0 + s0 * g.adj(a0)) * b0
    + 4 * (s0 * a0 + a0 * s0) * g.adj(b0)
)
assert_field_close(
    n3.value.value.gradient, ref_t, 1e-11, "quartic norm 3rd derivative"
)


# --- trigonometric action: exercises the rev-AD transform ops (sin/cos) in a
# nested setting.  For real data the exact derivatives of S = sum cos(s) are
# -sin, -cos*a, sin*a*b element-wise.  (For complex data the 3rd derivative
# carries the framework's contraction convention adj(a)*b instead of a*b.)
g.message("cos action: exact derivatives (nested transform ops)")

s_r = g.complex(grid)
rng_l.normal(s_r)
a_r = g.complex(grid)
rng_l.normal(a_r)
b_r = g.complex(grid)
rng_l.normal(b_r)
nar = rad.node(a_r, with_gradient=False)
nbr = rad.node(b_r, with_gradient=False)


def cos_action(n):
    return g.sum(g.component.cos(n))


nct = rad.node(s_r)
cos_action(nct)()
assert_field_close(nct.gradient, -g.component.sin(s_r), 1e-13, "cos action dS/ds")

n2ct = rad.node(rad.node(s_r))
cos_action(n2ct)()
c2ct = g.inner_product(nar, n2ct.gradient)
c2ct()
assert_field_close(n2ct.value.gradient, -g.component.cos(s_r) * a_r, 1e-12, "cos action HVP")

n3ct = rad.node(rad.node(rad.node(s_r)))
cos_action(n3ct)()
c3ct = g.inner_product(nar, n3ct.gradient)
c3ct()
c3ctb = g.inner_product(nbr, n3ct.value.gradient)
c3ctb()
assert_field_close(
    n3ct.value.value.gradient,
    g.component.sin(s_r) * a_r * b_r,
    1e-11,
    "cos action 3rd derivative",
)

# --- the contraction is symmetric in its arguments: the reversed slots must
# give the same (real) values and derivatives; a plain operand is promoted
# to a constant node ---
g.message("inner_product symmetry (reversed contraction)")

n2s = rad.node(rad.node(s_r))
cos_action(n2s)()
c2s = g.inner_product(n2s.gradient, nar)
c2s()
assert_close(c2s.value, c2ct.value, 1e-13, "reversed HVP value")
assert_field_close(
    n2s.value.gradient, -g.component.cos(s_r) * a_r, 1e-12, "reversed HVP field"
)

n3s = rad.node(rad.node(rad.node(s_r)))
cos_action(n3s)()
c3s = g.inner_product(n3s.gradient, nar)
c3s()
c3sb = g.inner_product(n3s.value.gradient, nbr)
c3sb()
assert_field_close(
    n3s.value.value.gradient,
    g.component.sin(s_r) * a_r * b_r,
    1e-11,
    "reversed 3rd derivative",
)

n2m = rad.node(rad.node(s_r))
cos_action(n2m)()
c2m = g.inner_product(n2m.gradient, a_r)
c2m()
assert_close(
    c2m.value, c2ct.value, 1e-13, "mixed-argument contraction value"
)

n2p = rad.node(rad.node(s_r))
cos_action(n2p)()
c2p = g.inner_product(a_r, n2p.gradient)
c2p()
assert_close(
    c2p.value, c2ct.value, 1e-13, "plain-first contraction value"
)


# --- second action with a cshift, checked via finite differences ---
g.message("hopping + quartic action: finite-difference checks")
rho = 0.3


def hop_action(s):
    C = s * g.adj(s)
    return (g.sum(C * C) + rho * g.sum(g.adj(s) * g.cshift(s, 0, 1))).real


def hop_first(s):
    nn = rad.node(s)
    hop_action(nn)()
    return nn.gradient


def hop_hvp(s, a_node):
    nn = rad.node(rad.node(s))
    hop_action(nn)()
    c = g.inner_product(a_node, nn.gradient)
    c()
    return nn.value.gradient


def hop_third(s, a_node, b_node):
    nn = rad.node(rad.node(rad.node(s)))
    hop_action(nn)()
    c = g.inner_product(a_node, nn.gradient)
    c()
    c2 = g.inner_product(b_node, nn.value.gradient)
    c2()
    return nn.value.value.gradient


# 1st derivative vs finite difference of the action (ad.py pattern:
# real direction, imag part perturbed by 1j)
n = rad.node(s0)
cc = (
    g.sum(n * g.adj(n) * n * g.adj(n)) + rho * g.sum(g.adj(n) * g.cshift(n, 0, 1))
)
for ig, part in [(1.0, lambda x: x.real), (1.0j, lambda x: x.imag)]:
    cc(initial_gradient=ig)
    eps = 1e-6
    lt = rng_l.normal(g.real(grid))
    n.value = g(s0 + lt * eps)
    v1 = part(cc(with_gradients=False))
    n.value = g(s0 - lt * eps)
    v2 = part(cc(with_gradients=False))
    n.value = g(s0)
    num_result = (v1 - v2) / eps / 2.0
    ad_result = g.inner_product(lt, n.gradient).real
    err = abs(num_result - ad_result) / (abs(num_result) + abs(ad_result) + 1)
    g.message(f"hop 1st derivative real (ig={ig}): {err}")
    assert err < 1e-4, "hop 1st derivative real"

    n.value = g(s0 + lt * eps * 1.0j)
    v1 = part(cc(with_gradients=False))
    n.value = g(s0 - lt * eps * 1.0j)
    v2 = part(cc(with_gradients=False))
    n.value = g(s0)
    num_result = (v1 - v2) / eps / 2.0
    ad_result = g.inner_product(lt, n.gradient).imag
    err = abs(num_result - ad_result) / (abs(num_result) + abs(ad_result) + 1)
    g.message(f"hop 1st derivative imag (ig={ig}): {err}")
    assert err < 1e-4, "hop 1st derivative imag"

# 2nd derivative: HVP (direction b) vs finite difference of the 1st derivative
# in the same direction
eps = 1e-5
hvp_ad = hop_hvp(s0, nb)
hvp_fd = (hop_first(g(s0 + eps * b0)) - hop_first(g(s0 - eps * b0))) / (2 * eps)
assert_field_close(hvp_ad, hvp_fd, 1e-5, "hop HVP vs FD of 1st derivative")

# 3rd derivative vs finite difference of the HVP field
t3_ad = hop_third(s0, na, nb)
t3_fd = (hop_hvp(g(s0 + eps * b0), na) - hop_hvp(g(s0 - eps * b0), na)) / (2 * eps)
assert_field_close(t3_ad, t3_fd, 1e-5, "hop 3rd derivative vs FD of HVP")

# symmetric scalar cross-check: d^3 S(s + t b)/dt^3 vs a 4-point 3rd-order
# finite difference of the action itself
h = 1e-3


def f_of_t(t):
    return hop_action(g(s0 + t * b0))


fd = (f_of_t(2 * h) - 2 * f_of_t(h) + 2 * f_of_t(-h) - f_of_t(-2 * h)) / (
    2 * h**3
)
t3_b = hop_third(s0, nb, nb)
ad = g.inner_product(b0, t3_b).real
assert_close(ad, fd, 1e-5, "hop d3S/dt^3 vs FD of action")

# --- division by a scalar constant: the only division the C++ core supports
# (field / scalar == field * (1/scalar)).  node / node and scalar / node
# would need a pointwise-reciprocal kernel (A2), so they are not tested here.
# Checked to 2nd order via finite differences, reusing the graph on modified
# leaf values (the node_differentiable_functional path) for the 1st derivative.
g.message("division by a scalar: finite-difference checks")
cdiv = 2.0 + 0.5j


def div_action(n):
    q = n / cdiv
    C = q * g.adj(q)
    return g.sum(C * C)


def div_first(s):
    nn = rad.node(s)
    div_action(nn)()
    return nn.gradient


def div_hvp(s, a_node):
    nn = rad.node(rad.node(s))
    div_action(nn)()
    c = g.inner_product(a_node, nn.gradient)
    c()
    return nn.value.gradient


n = rad.node(s0)
dact = div_action(n)
for ig, part in [(1.0, lambda x: x.real), (1.0j, lambda x: x.imag)]:
    dact(initial_gradient=ig)
    eps = 1e-6
    lt = rng_l.normal(g.real(grid))
    n.value = g(s0 + lt * eps)
    v1 = part(dact(with_gradients=False))
    n.value = g(s0 - lt * eps)
    v2 = part(dact(with_gradients=False))
    n.value = g(s0)
    num_result = (v1 - v2) / eps / 2.0
    ad_result = g.inner_product(lt, n.gradient).real
    err = abs(num_result - ad_result) / (abs(num_result) + abs(ad_result) + 1)
    g.message(f"div 1st derivative real (ig={ig}): {err}")
    assert err < 1e-4, "div 1st derivative real"

# 2nd derivative: HVP (direction b) vs finite difference of the 1st derivative
eps = 1e-5
hvp_ad = div_hvp(s0, nb)
hvp_fd = (div_first(g(s0 + eps * b0)) - div_first(g(s0 - eps * b0))) / (2 * eps)
assert_field_close(hvp_ad, hvp_fd, 1e-5, "div HVP vs FD of 1st derivative")

# --- division by a gradient-carrying COMPLEX scalar denominator: exercises
# the to-y backprop (dS/dy = -sum(x)/y^2, conjugate-linear -> a scalar flow).
# x is a constant, so S = X/y with X = sum(x); checked against finite
# differences of the scalar action at both initial gradients.
g.message("division by a complex scalar denominator (to-y backprop)")
xd = g.complex(grid)
rng_l.cnormal(xd)
yd0 = 1.5 + 0.7j
xnd = rad.node(xd, with_gradient=False)


def S_div(y):
    return g.sum(xd / y)  # lattice / scalar -> lattice; sum -> scalar


def dy_action(yn):
    return g.sum(xnd / yn)


eps = 1e-6
dS_dy = (S_div(yd0 + eps) - S_div(yd0 - eps)) / (2 * eps)
for ig, part in [(1.0, lambda x: x.real), (1.0j, lambda x: x.imag)]:
    ynd = rad.node(yd0)
    dyact = dy_action(ynd)
    dyact(initial_gradient=ig)
    num_result = part(dS_dy)
    ad_result = ynd.gradient.real
    err = abs(num_result - ad_result) / (abs(num_result) + abs(ad_result) + 1)
    g.message(f"div to-y 1st derivative (ig={ig}): {err}")
    assert err < 1e-4, f"div to-y 1st derivative (ig={ig})"

# --- integer power on lattice data: the C++ core has no power op, so
# node ** int is built from repeated multiplication (A1).  Checked to 2nd
# order via finite differences (S = sum n^4, a complex action so both the
# real and imaginary parts are exercised).
g.message("integer power (lattice): finite-difference checks")


def pow_action(n):
    return g.sum(n ** 4)


def pow_first(s):
    nn = rad.node(s)
    pow_action(nn)()
    return nn.gradient


def pow_hvp(s, a_node):
    nn = rad.node(rad.node(s))
    pow_action(nn)()
    c = g.inner_product(a_node, nn.gradient)
    c()
    return nn.value.gradient


n = rad.node(s0)
pact = pow_action(n)
for ig, part in [(1.0, lambda x: x.real), (1.0j, lambda x: x.imag)]:
    pact(initial_gradient=ig)
    eps = 1e-6
    lt = rng_l.normal(g.real(grid))
    n.value = g(s0 + lt * eps)
    v1 = part(pact(with_gradients=False))
    n.value = g(s0 - lt * eps)
    v2 = part(pact(with_gradients=False))
    n.value = g(s0)
    num_result = (v1 - v2) / eps / 2.0
    ad_result = g.inner_product(lt, n.gradient).real
    err = abs(num_result - ad_result) / (abs(num_result) + abs(ad_result) + 1)
    g.message(f"pow 1st derivative real (ig={ig}): {err}")
    assert err < 1e-4, "pow 1st derivative real"

# 2nd derivative: HVP (direction b) vs finite difference of the 1st derivative
eps = 1e-5
hvp_ad = pow_hvp(s0, nb)
hvp_fd = (pow_first(g(s0 + eps * b0)) - pow_first(g(s0 - eps * b0))) / (2 * eps)
assert_field_close(hvp_ad, hvp_fd, 1e-5, "pow HVP vs FD of 1st derivative")

# --- conjugate-linear gradient convention: every backprop applies g.adj to
# its cofactor (matching __mul__), so gradient = conj(Wirtinger d/dx).  This
# is only observable for COMPLEX data and in the imaginary part (ig=1.0j);
# real data and real-valued actions are unaffected (adj = identity).
g.message("conjugate-linear convention: complex-data checks")

# (a) scalar **: n**2 and n*n are the same function, so their gradients must
# agree; both are conjugate-linear (2*adj(x0)), not holomorphic (2*x0).
x0c = 0.7 + 0.4j
np1 = rad.node(x0c); (np1**2)()
np2 = rad.node(x0c); (np2*np2)()
assert_close(np1.gradient, 2 * g.adj(x0c), 1e-14, "scalar ** grad (conj-linear)")
assert_close(np1.gradient, np2.gradient, 1e-14, "scalar ** == scalar * grad")

# (b) sin on a complex lattice: a complex-valued action, checked against finite
# differences at both initial gradients (ig=1.0j is the distinguishing case)
def conv_action(n):
    return g.sum(g.component.sin(n))


n = rad.node(s0)
cact = conv_action(n)
for ig, part in [(1.0, lambda x: x.real), (1.0j, lambda x: x.imag)]:
    cact(initial_gradient=ig)
    eps = 1e-6
    lt = rng_l.normal(g.real(grid))
    n.value = g(s0 + lt * eps)
    v1 = part(cact(with_gradients=False))
    n.value = g(s0 - lt * eps)
    v2 = part(cact(with_gradients=False))
    n.value = g(s0)
    num_result = (v1 - v2) / eps / 2.0
    ad_result = g.inner_product(lt, n.gradient).real
    err = abs(num_result - ad_result) / (abs(num_result) + abs(ad_result) + 1)
    g.message(f"convention sin 1st derivative (ig={ig}): {err}")
    assert err < 1e-4, f"convention sin 1st derivative (ig={ig})"

#####################################
# stage 2: SU(3) gauge-field HVP (2nd derivative)
#
# Production convention (applications/hmc/hessian.py):  for an algebra
# direction dA the group flow is U(t) = compose(exp(t dA), U); the HVP
# HVP(dA) = nnU.value.gradient after evaluating
#     c = sum_mu group.inner_product(nnU[mu].gradient, nA[mu]);  c()
# and must satisfy the Taylor identity
#     S(U(t)) = S(U) + t IP(F, dA) + t^2/2 IP(dA, HVP(dA)) + O(t^3)
# with F the 1st derivative.  The covariant Hessian
#     covHVP = HVP - 0.5j (F dA - dA F)
# is symmetric:  IP(dA2, covHVP(dA)) = IP(dA, covHVP(dA2)).

gridg = g.grid([4, 4, 4, 4], g.double)
rngg = g.random("gauge_test")
Ug = g.qcd.gauge.random(gridg, rngg, scale=2.0)
action_g = g.qcd.gauge.action.differentiable_iwasaki(2.95)

nnUg = [rad.node(rad.node(u)) for u in Ug]
nAg = [rad.node(g.group.cartesian(u)) for u in Ug]
action_g(nnUg)()


def gauge_hvp(dA):
    for mu in range(4):
        nAg[mu].value @= dA[mu]
    c = sum(g.group.inner_product(nnUg[mu].gradient, nAg[mu]) for mu in range(4))
    c()
    return [resolve_value(nnUg[mu].value.gradient) for mu in range(4)]


dA = rngg.normal_element(g.group.cartesian(Ug))
dA2 = g.random("gauge_test2").normal_element(g.group.cartesian(Ug))
H_dA = gauge_hvp(dA)
H_dA2 = gauge_hvp(dA2)

# 1st derivative (flat) for the Taylor test
F = g.qcd.gauge.action.iwasaki(2.95).gradient(Ug, Ug)

# Taylor identity: 2nd-order term
eps = 1e-4
Ue = [g(g.group.compose(g(eps * dA[mu]), Ug[mu])) for mu in range(4)]
Um = [g(g.group.compose(g(-eps * dA[mu]), Ug[mu])) for mu in range(4)]
a0 = g(action_g(Ug))
a1 = g(action_g(Ue))
am = g(action_g(Um))
F_dA = sum(g.group.inner_product(F[mu], dA[mu]) for mu in range(4))
dA_H_dA = sum(g.group.inner_product(dA[mu], H_dA[mu]) for mu in range(4))
# central 2nd-order FD of the action along the group flow
d2S_fd = (a1 - 2 * a0 + am) / eps**2
err = abs(d2S_fd - dA_H_dA) / (abs(d2S_fd) + abs(dA_H_dA) + 1e-30)
g.message(f"gauge HVP bilinear vs 2nd-order FD: {err}")
assert err < 1e-4, "gauge HVP bilinear vs 2nd-order FD"
# Taylor 3rd-order residual (expect O(eps) after dividing by eps^2)
res = abs(a1 - a0 - eps * F_dA - 0.5 * eps**2 * dA_H_dA) / (
    eps**2 * (abs(dA_H_dA) + 1e-30)
)
g.message(f"gauge Taylor 2nd-order relative residual: {res}")
assert res < 1e-3, "gauge Taylor 2nd-order relative residual"

# covariant Hessian symmetry
covH_dA = [
    H_dA[mu] - 0.5j * (F[mu] * dA[mu] - dA[mu] * F[mu]) for mu in range(4)
]
covH_dA2 = [
    H_dA2[mu] - 0.5j * (F[mu] * dA2[mu] - dA2[mu] * F[mu]) for mu in range(4)
]
ip1 = sum(g.group.inner_product(dA2[mu], covH_dA[mu]) for mu in range(4))
ip2 = sum(g.group.inner_product(dA[mu], covH_dA2[mu]) for mu in range(4))
err = abs(ip1 - ip2) / (abs(ip1) + abs(ip2) + 1e-30)
g.message(f"gauge covariant Hessian symmetry: {err}")
assert err < 1e-6, "gauge covariant Hessian symmetry"

#####################################
# stage 3: 3rd derivative of the gauge action (gradient of the Hessian)
#
# With nnnU = node(node(node(U))), three reverse passes give
#   G3[mu] = nnnU[mu].value.value.gradient
# the gradient with respect to the gauge field of the Hessian bilinear
# form d^2 S(A, B):  IP(C, G3) = d^3 S(C, A, B).  Cross-checks:
#  - FD of the HVP field along the group flow in direction A:
#    [HVP_A(U + eps A) - HVP_A(U - eps A)] / (2 eps)
#  - 4-point 3rd-order FD of the action for A = B = C:
#    IP(A, G3) = f'''(0),  f(t) = S(compose(exp(t A), U))

nnnUg = [rad.node(rad.node(rad.node(u))) for u in Ug]
action_g(nnnUg)()
nA3 = [rad.node(g.group.cartesian(u)) for u in Ug]
nB3 = [rad.node(g.group.cartesian(u)) for u in Ug]
for mu in range(4):
    nA3[mu].value @= dA[mu]
    nB3[mu].value @= dA[mu]
cb3 = sum(
    g.group.inner_product(nnnUg[mu].gradient, nB3[mu]) for mu in range(4)
)
cb3()
ca3 = sum(
    g.group.inner_product(nnnUg[mu].value.gradient, nA3[mu]) for mu in range(4)
)
ca3()
G3 = [resolve_value(nnnUg[mu].value.value.gradient) for mu in range(4)]
dA_d2S_dA = sum(g.group.inner_product(dA[mu], G3[mu]) for mu in range(4))

# FD of the HVP field along the group flow


def gauge_hvp_field(Ucfg, dAd):
    nnU = [rad.node(rad.node(u)) for u in Ucfg]
    nD = [rad.node(g.group.cartesian(u)) for u in Ucfg]
    action_g(nnU)()
    for mu in range(4):
        nD[mu].value @= dAd[mu]
    c = sum(g.group.inner_product(nnU[mu].gradient, nD[mu]) for mu in range(4))
    c()
    return [resolve_value(nnU[mu].value.gradient) for mu in range(4)]


eps3 = 1e-4
Ue3 = [g(g.group.compose(g(eps3 * dA[mu]), Ug[mu])) for mu in range(4)]
Um3 = [g(g.group.compose(g(-eps3 * dA[mu]), Ug[mu])) for mu in range(4)]
Hvp_p = gauge_hvp_field(Ue3, dA)
Hvp_m = gauge_hvp_field(Um3, dA)
G3_ref = [(Hvp_p[mu] - Hvp_m[mu]) / (2 * eps3) for mu in range(4)]
err = g.norm2(
    sum(g.group.inner_product(dA[mu], G3[mu] - G3_ref[mu]) for mu in range(4))
) ** 0.5 / g.norm2(
    sum(g.group.inner_product(dA[mu], G3_ref[mu]) for mu in range(4))
) ** 0.5
g.message(f"gauge 3rd derivative vs FD of HVP: {err}")
assert err < 1e-4, "gauge 3rd derivative vs FD of HVP"

# 4-point 3rd-order FD of the action for A = B = C = dA
def f3_of_t(t):
    Ut = [g(g.group.compose(g(t * dA[mu]), Ug[mu])) for mu in range(4)]
    return g(action_g(Ut))


h3 = 1e-3
fd3 = (f3_of_t(2 * h3) - 2 * f3_of_t(h3) + 2 * f3_of_t(-h3) - f3_of_t(-2 * h3)) / (
    2 * h3**3
)
err = abs(dA_d2S_dA - fd3) / (abs(dA_d2S_dA) + abs(fd3) + 1e-30)
g.message(f"gauge d3S(A,A,A) vs 4-point FD of action: {err}")
assert err < 1e-3, "gauge d3S(A,A,A) vs 4-point FD of action"
