#!/usr/bin/env python3
#
# Authors: Christoph Lehner
# Acknowledgements: Qwen 3.8 27b in Pi Coding Agent
#
# Desc.: Higher-order (2nd and 3rd) derivatives in the reverse-accumulation
#        AD framework: recorded reverse passes (create_graph=True).
#
# A reverse pass with create_graph=True records the backward: the leaf's
# .gradient is then a (lazy) node graph for dS/dx over the same leaves.
# Contracting it with a direction gives an ordinary scalar node, whose
# reverse pass deposits the next derivative into the leaf's .gradient; the
# order is chosen per pass, e.g. for n = node(x):
#
#   S(n).backward(create_graph=True)                -> n.gradient: graph for dS/dx
#   inner_product(a, n.gradient).backward(create_graph=True)
#                                                    -> n.gradient: graph for d2S/dx2 a
#   inner_product(b, n.gradient).backward()          -> n.gradient = d3S/dx3 a b
#
# Conventions:
#  - the contraction must be linear in the gradient argument.  For lattice
#    nodes, g.inner_product(direction, gradient) puts the gradient in the
#    linear (second) slot; for scalar nodes use g.adj(direction) * gradient
#    (node __mul__ is adjoint-linear in its first argument).  Directions are
#    plain values (constants).
#  - for SU(N) gauge fields, contract with g.group.inner_product as in
#    applications/hmc/hessian.py.
#
import gpt as g
from gpt.ad.reverse.util import is_node, value_of

rng = g.random("test")
rad = g.ad.reverse


def resolve_value(x):
    # the plain value of a result that may be a (lazy) node, e.g. a recorded
    # gradient
    if is_node(x):
        x = value_of(x)
    return g(x) if isinstance(x, g.expr) else x


def hvp_of(S, x, a, contract=None, **leaf):
    # d/dx <a, dS/dx>: a recorded pass and a reverse pass of the contraction
    contract = contract or g.inner_product
    n = rad.node(x, **leaf)
    S(n).backward(create_graph=True)
    contract(a, n.gradient).backward()
    return n.gradient


def third_of(S, x, a, b, contract=None):
    # d/dx <b, d/dx <a, dS/dx>>: two recorded passes and a reverse pass
    contract = contract or g.inner_product
    n = rad.node(x)
    S(n).backward(create_graph=True)
    contract(a, n.gradient).backward(create_graph=True)
    contract(b, n.gradient).backward()
    return n.gradient


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

    def S_scalar(m):
        return m**4 + 3.0 * m**2

    def scalar_contract(d, gr):
        return g.adj(d) * gr

    assert_close(
        hvp_of(S_scalar, x0, a, scalar_contract), (12 * x0**2 + 6) * a, 1e-13, "scalar d2S/dx2 * a"
    )
    assert_close(
        third_of(S_scalar, x0, a, b, scalar_contract), 24 * x0 * a * b, 1e-12, "scalar d3S/dx3 * a * b"
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

ref_h = 4 * C0 * a0 + 4 * (g.adj(s0) * a0 + s0 * g.adj(a0)) * s0
assert_field_close(hvp_of(quartic_norm, s0, a0), ref_h, 1e-12, "quartic norm HVP")

ref_t = (
    4 * (2 * g.adj(s0) * a0 + g.adj(a0) * s0 + s0 * g.adj(a0)) * b0
    + 4 * (s0 * a0 + a0 * s0) * g.adj(b0)
)
assert_field_close(
    third_of(quartic_norm, s0, a0, b0), ref_t, 1e-11, "quartic norm 3rd derivative"
)

# the recorded gradient is a graph over the same leaf: evaluating it gives
# the 1st derivative
n = rad.node(s0)
quartic_norm(n).backward(create_graph=True)
assert is_node(n.gradient)
assert_field_close(n.gradient, 4 * C0 * s0, 1e-13, "quartic norm recorded dS/ds")


# --- trigonometric action: exercises the rev-AD transform ops (sin/cos) in
# recorded passes.  For real data the exact derivatives of S = sum cos(s) are
# -sin, -cos*a, sin*a*b element-wise.  (For complex data the 3rd derivative
# carries the framework's contraction convention adj(a)*b instead of a*b.)
g.message("cos action: exact derivatives (recorded transform ops)")

s_r = g.complex(grid)
rng_l.normal(s_r)
a_r = g.complex(grid)
rng_l.normal(a_r)
b_r = g.complex(grid)
rng_l.normal(b_r)


def cos_action(n):
    return g.sum(g.component.cos(n))


nct = rad.node(s_r)
cos_action(nct)()
assert_field_close(nct.gradient, -g.component.sin(s_r), 1e-13, "cos action dS/ds")

n2ct = rad.node(s_r)
cos_action(n2ct).backward(create_graph=True)
c2ct = g.inner_product(a_r, n2ct.gradient)
c2ct_value = c2ct()
assert_field_close(n2ct.gradient, -g.component.cos(s_r) * a_r, 1e-12, "cos action HVP")

assert_field_close(
    third_of(cos_action, s_r, a_r, b_r),
    g.component.sin(s_r) * a_r * b_r,
    1e-11,
    "cos action 3rd derivative",
)

# --- the contraction is symmetric in its arguments: the reversed slots must
# give the same (real) values and derivatives; a plain operand is promoted
# to a constant node (either slot) ---
g.message("inner_product symmetry (reversed contraction)")


def reversed_contract(d, gr):
    return g.inner_product(gr, d)


n2s = rad.node(s_r)
cos_action(n2s).backward(create_graph=True)
c2s = g.inner_product(n2s.gradient, a_r)
assert_close(c2s(), c2ct_value, 1e-13, "reversed HVP value")
assert_field_close(
    n2s.gradient, -g.component.cos(s_r) * a_r, 1e-12, "reversed HVP field"
)

assert_field_close(
    third_of(cos_action, s_r, a_r, b_r, reversed_contract),
    g.component.sin(s_r) * a_r * b_r,
    1e-11,
    "reversed 3rd derivative",
)

n2m = rad.node(s_r)
cos_action(n2m).backward(create_graph=True)
c2m = g.inner_product(n2m.gradient, rad.node(a_r, with_gradient=False))
assert_close(
    c2m(), c2ct_value, 1e-13, "constant-node contraction value"
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


def hop_hvp(s, a):
    return hvp_of(hop_action, s, a)


def hop_third(s, a, b):
    return third_of(hop_action, s, a, b)


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
hvp_ad = hop_hvp(s0, b0)
hvp_fd = (hop_first(g(s0 + eps * b0)) - hop_first(g(s0 - eps * b0))) / (2 * eps)
assert_field_close(hvp_ad, hvp_fd, 1e-5, "hop HVP vs FD of 1st derivative")

# 3rd derivative vs finite difference of the HVP field
t3_ad = hop_third(s0, a0, b0)
t3_fd = (hop_hvp(g(s0 + eps * b0), a0) - hop_hvp(g(s0 - eps * b0), a0)) / (2 * eps)
assert_field_close(t3_ad, t3_fd, 1e-5, "hop 3rd derivative vs FD of HVP")

# symmetric scalar cross-check: d^3 S(s + t b)/dt^3 vs a 4-point 3rd-order
# finite difference of the action itself
h = 1e-3


def f_of_t(t):
    return hop_action(g(s0 + t * b0))


fd = (f_of_t(2 * h) - 2 * f_of_t(h) + 2 * f_of_t(-h) - f_of_t(-2 * h)) / (
    2 * h**3
)
t3_b = hop_third(s0, b0, b0)
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


def div_hvp(s, a):
    return hvp_of(div_action, s, a)


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
hvp_ad = div_hvp(s0, b0)
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


def pow_hvp(s, a):
    return hvp_of(pow_action, s, a)


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
hvp_ad = pow_hvp(s0, b0)
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
# HVP(dA) = nU.gradient after
#     action(nU).backward(create_graph=True)            (recorded once)
#     c = sum_mu group.inner_product(gradient[mu], dA[mu]);  c.backward()
# (gradient: the recorded nU[mu].gradient) and must satisfy the Taylor identity
#     S(U(t)) = S(U) + t IP(F, dA) + t^2/2 IP(dA, HVP(dA)) + O(t^3)
# with F the 1st derivative.  The covariant Hessian
#     covHVP = HVP - 0.5j (F dA - dA F)
# is symmetric:  IP(dA2, covHVP(dA)) = IP(dA, covHVP(dA2)).

gridg = g.grid([4, 4, 4, 4], g.double)
rngg = g.random("gauge_test")
Ug = g.qcd.gauge.random(gridg, rngg, scale=2.0)
action_g = g.qcd.gauge.action.differentiable_iwasaki(2.95)

# the gradient graph is recorded once and contracted per direction
nUg = [rad.node(u) for u in Ug]
action_g(nUg).backward(create_graph=True)
grad_g = [x.gradient for x in nUg]


def gauge_hvp(dA):
    c = sum(g.group.inner_product(grad_g[mu], dA[mu]) for mu in range(4))
    c.backward()
    return [nUg[mu].gradient for mu in range(4)]


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
# Two recorded passes and a reverse pass give G3[mu] = nU[mu].gradient,
# the gradient with respect to the gauge field of the Hessian bilinear
# form d^2 S(A, B):  IP(C, G3) = d^3 S(C, A, B).  Cross-checks:
#  - FD of the HVP field along the group flow in direction A:
#    [HVP_A(U + eps A) - HVP_A(U - eps A)] / (2 eps)
#  - 4-point 3rd-order FD of the action for A = B = C:
#    IP(A, G3) = f'''(0),  f(t) = S(compose(exp(t A), U))

nU3 = [rad.node(u) for u in Ug]
action_g(nU3).backward(create_graph=True)
sum(g.group.inner_product(nU3[mu].gradient, dA[mu]) for mu in range(4)).backward(
    create_graph=True
)
sum(g.group.inner_product(nU3[mu].gradient, dA[mu]) for mu in range(4)).backward()
G3 = [nU3[mu].gradient for mu in range(4)]
dA_d2S_dA = sum(g.group.inner_product(dA[mu], G3[mu]) for mu in range(4))

# FD of the HVP field along the group flow


def gauge_hvp_field(Ucfg, dAd):
    nU = [rad.node(u) for u in Ucfg]
    action_g(nU).backward(create_graph=True)
    sum(g.group.inner_product(nU[mu].gradient, dAd[mu]) for mu in range(4)).backward()
    return [nU[mu].gradient for mu in range(4)]


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

#####################################
# stage 4: 3rd derivative via the functional force mechanism
#
# The Hessian bilinear form d^2 S(A, B) = IP(B, H(U) A) is itself a
# differentiable scalar function of the gauge field.  With two recorded
# passes the HVP H(U) A is a lazy node over the leaves, so the bilinear form
#     S = sum_mu group.inner_product(B[mu], HVP_A[mu])
# is a node (a compute graph over the gauge field) rather than a plain
# scalar.  Its force w.r.t. U -- via node_differentiable_functional
# .gradient, the same reusable-graph force mechanism the production action
# uses -- is the 3rd derivative:
#     G3_fun = S.functional(*nU).gradient(U, U)
# The single reverse pass through S both evaluates the bilinear form and
# accumulates this force, so it must agree exactly with stage 3 (G3 /
# dA_d2S_dA), which builds the same graph.  The functional swaps the leaf
# values, so the recorded graph is re-evaluated at other gauge fields
# (assert_gradient_error below) without re-recording.

nU4 = [rad.node(g.copy(u)) for u in Ug]
action_g(nU4).backward(create_graph=True)
sum(g.group.inner_product(nU4[mu].gradient, dA[mu]) for mu in range(4)).backward(
    create_graph=True
)
HVP_A4 = [nU4[mu].gradient for mu in range(4)]
# the Hessian bilinear form as a node; not evaluated here -- the functional's
# gradient() below evaluates it and takes its force in one pass
S4 = sum(g.group.inner_product(dA[mu], HVP_A4[mu]) for mu in range(4))
f4 = S4.functional(*nU4)
G3_fun = f4.gradient(Ug, Ug)
dA_d2S_dA_fun = sum(g.group.inner_product(dA[mu], G3_fun[mu]) for mu in range(4))

err = abs(dA_d2S_dA_fun - dA_d2S_dA) / (
    abs(dA_d2S_dA_fun) + abs(dA_d2S_dA) + 1e-30
)
g.message(f"gauge 3rd derivative functional vs recorded passes (stage 3): {err}")
assert err < 1e-14, "gauge 3rd derivative functional vs recorded passes (stage 3)"

for mu in range(4):
    err = g.norm2(G3_fun[mu] - G3[mu]) / g.norm2(G3[mu])
    g.message(f"gauge 3rd derivative functional field mu={mu} vs stage 3: {err}")
    assert err < 1e-14, f"gauge 3rd derivative functional field mu={mu} vs stage 3"

# the Hessian bilinear form is a genuine differentiable action of U, so the
# standard force-mechanism FD cross-check applies to its 3rd-derivative force:
# assert_gradient_error compares the AD gradient (the 3rd derivative) against
# a 4th-order FD of the bilinear form along the group flow, and checks that the
# force lives in the cartesian (Lie algebra) representation.  (Unlike the gauge
# *action* functional, whose gradient is the 1st derivative and is covered by
# ad.py, this functional's gradient is the 3rd derivative.)
rng_hf = g.random("gauge_test_hessian_force")
f4.assert_gradient_error(rng_hf, Ug, Ug, 1e-3, 1e-6)

#####################################
# group_inner_product symmetry (C2)
#
# The gauge group contraction must be symmetric in its arguments: the
# reversed slots give the same (real) value and the same gradient, and a
# plain operand is promoted to a constant node.  Checked on a gauge
# Cartesian field (the otype that actually has an inner_product).
g.message("group_inner_product symmetry (reversed contraction)")
c0g = g.group.cartesian(Ug[0])
ag = rngg.normal_element(g.group.cartesian(Ug))[0]

n1g = rad.node(c0g)
S1g = g.group.inner_product(n1g, ag)  # node first
S1g()
n2g = rad.node(c0g)
S2g = g.group.inner_product(ag, n2g)  # plain first
S2g()
assert_close(
    resolve_value(S1g), resolve_value(S2g), 1e-13, "group inner_product symmetry value"
)
assert_field_close(
    resolve_value(n1g.gradient),
    resolve_value(n2g.gradient),
    1e-13,
    "group inner_product symmetry grad",
)

#####################################
# foundational passthrough ops: where (mask routing) and astype (type cast)
#
# where routes the constant "yes"/"no" branch per mask and its backprop
# routes the flow the same way; astype is a type cast that passes the flow
# straight through.  Both are checked to exact 1st order on constant all-1 /
# all-0 masks, and in recorded passes, where the flow (a node) must route to
# the rev-AD op (via nodify) instead of the plain foundation (which cannot
# build a lattice from a node).  The 2nd derivative (HVP) through where is
# checked against the plain quartic-norm HVP (reusing quartic_norm from
# stage 1).
g.message("foundational passthrough ops: where / astype")

w_s = g.complex(grid)
rng_l.cnormal(w_s)
w_t = g.complex(grid)
rng_l.cnormal(w_t)
ident4 = g.identity(g.complex(grid))
zeros4 = w_s * 0.0

# --- where: 1st-order mask routing (constant all-1 / all-0 masks) ---
Aw = rad.node(w_s)
Bw = rad.node(w_t)
g.sum(g.where(ident4, Aw, Bw))()  # all-1: selects the "yes" branch
assert_field_close(Aw.gradient, ident4, 1e-14, "where all-1 yes gradient = 1")
assert g.norm2(Bw.gradient) < 1e-14, "where all-1 no gradient = 0"

Aw = rad.node(w_s)
Bw = rad.node(w_t)
g.sum(g.where(zeros4, Aw, Bw))()  # all-0: selects the "no" branch
assert g.norm2(Aw.gradient) < 1e-14, "where all-0 yes gradient = 0"
assert_field_close(Bw.gradient, ident4, 1e-14, "where all-0 no gradient = 1")

# --- where: recorded passes ---
Aw = rad.node(w_s)
Bw = rad.node(w_t)
g.sum(g.where(ident4, Aw, Bw)).backward(create_graph=True)  # all-1: recorded gradients
assert_field_close(Aw.gradient, ident4, 1e-14, "where recorded yes 1st gradient")
assert g.norm2(resolve_value(Bw.gradient)) < 1e-14, "where recorded no gradient = 0"

# 2nd derivative (HVP) through where: all-1 mask == the "yes" branch, so
# S = sum |where(mask, A, B)|^4 = sum |A|^4 and the HVP is the plain
# quartic-norm HVP of w_s.
a0w = rng_l.cnormal(g.complex(grid))
C0w = w_s * g.adj(w_s)
ref_h = 4 * C0w * a0w + 4 * (g.adj(w_s) * a0w + w_s * g.adj(a0w)) * w_s
assert_field_close(
    hvp_of(lambda A: quartic_norm(g.where(ident4, A, rad.node(w_t))), w_s, a0w),
    ref_h,
    1e-12,
    "where recorded quartic HVP",
)

# all-0 mask: the flow routes to the "no" branch
C0t = w_t * g.adj(w_t)
ref_ht = 4 * C0t * a0w + 4 * (g.adj(w_t) * a0w + w_t * g.adj(a0w)) * w_t
assert_field_close(
    hvp_of(lambda B: quartic_norm(g.where(zeros4, rad.node(w_s), B)), w_t, a0w),
    ref_ht,
    1e-12,
    "where recorded no-branch quartic HVP",
)

# --- astype: type-cast passthrough (complex -> real) ---
real_ot = g.real(grid).otype
a_s = g.complex(grid)
rng_l.cnormal(a_s)

Aa = rad.node(a_s)
g.sum(g.astype(Aa, real_ot))()
assert_field_close(Aa.gradient, ident4, 1e-14, "astype 1st gradient = 1")

A2a = rad.node(a_s)
g.sum(g.astype(A2a, real_ot)).backward(create_graph=True)  # recorded
assert_field_close(A2a.gradient, ident4, 1e-14, "astype recorded 1st gradient = 1")

#####################################
# adj op: the conjugate, exercised explicitly.  It also underlies the __mul__
# backprop (covered everywhere via the conjugate-linear convention), but the
# op itself was never tested directly.  The forward is the conjugate and the
# backprop is the double-adjoint (d(adj x)/dx = adj, so the flow back to x is
# conj(flow)).  The 1st derivative is exact (S = sum(adj n * m) with a
# constant m -> dS/dn = m); the 2nd derivative (HVP) of a nonlinear adj action
# is checked against finite differences.
g.message("adj op: explicit checks (also recorded)")

s_adj = g.complex(grid)
rng_l.cnormal(s_adj)
m_adj = g.complex(grid)
rng_l.cnormal(m_adj)
nm_adj = rad.node(m_adj, with_gradient=False)

n1a = rad.node(s_adj)
g.sum(g.adj(n1a) * nm_adj)()
assert_field_close(n1a.gradient, m_adj, 1e-14, "adj 1st gradient")

n2a = rad.node(s_adj)
g.sum(g.adj(n2a) * nm_adj).backward(create_graph=True)
assert_field_close(n2a.gradient, m_adj, 1e-14, "adj recorded 1st gradient")


def adj_action(n):
    return g.sum(g.adj(n) * n * n)


def adj_first(s_):
    nn = rad.node(s_)
    adj_action(nn)()
    return nn.gradient


a0_adj = rng_l.cnormal(g.complex(grid))
eps = 1e-5
hvp_ad = hvp_of(adj_action, s_adj, a0_adj)
hvp_fd = (adj_first(g(s_adj + eps * a0_adj)) - adj_first(g(s_adj - eps * a0_adj))) / (
    2 * eps
)
assert_field_close(hvp_ad, hvp_fd, 1e-5, "adj HVP vs FD of 1st derivative")

#####################################
# the values of nodes are plain values (or forward-AD series), never nodes:
# a node as the value of a leaf, or assigned as a value, is rejected
g.message("node values: a node as a value is rejected")

s_fun = g.complex(grid)
rng_l.cnormal(s_fun)
for attempt in ["construct", "assign"]:
    try:
        if attempt == "construct":
            rad.node(rad.node(s_fun))
        else:
            rad.node(s_fun).value = rad.node(s_fun)
        raise AssertionError(f"a node value was accepted ({attempt})")
    except ValueError as e:
        g.message(f"rejected ({attempt}): {e}")


# -----------------------------------------------------------------------------
# traceless (anti-)hermitian projections: single nodes whose backward is the
# projection of the flow, i.e. a projection node in a recorded pass.  HVP vs finite difference of the 1st derivative.  The mcolor leaves
# are additive (no conversion of their gradients to the group algebra).
# -----------------------------------------------------------------------------
g.message("traceless projections: HVP vs finite difference")
P = g.qcd.gauge.project
m0 = rng_l.cnormal(g.mcolor(grid))
mb = rng_l.cnormal(g.mcolor(grid))


def proj_action(n):
    q = P.traceless_anti_hermitian(n * n) + P.traceless_hermitian(n * g.adj(n))
    return g.sum(g.trace(q * q * n))


def proj_first(s):
    nn = rad.node(s, infinitesimal_to_cartesian=False)
    proj_action(nn)()
    return nn.gradient


eps = 1e-5
hvp_ad = hvp_of(proj_action, m0, mb, infinitesimal_to_cartesian=False)
hvp_fd = (proj_first(g(m0 + eps * mb)) - proj_first(g(m0 - eps * mb))) / (2 * eps)
assert_field_close(hvp_ad, hvp_fd, 1e-12, "projection HVP vs FD of 1st derivative")


#####################################
# real and imaginary parts in recorded passes: the Hessian-vector product of
# S = |f(z)^2 c - c|^2 (f = Re or Im) vs a difference of the gradient
c = rng.cnormal(g.complex(grid))
for op in ["real", "imag"]:
    f = getattr(g.component, op)
    z0 = rng.cnormal(g.complex(grid))
    v = rng.cnormal(g.complex(grid))

    def S(z):
        return g.norm2(f(z) * f(z) * c - c)

    hvp = hvp_of(S, z0, v)

    def first(z):
        n = rad.node(z)
        S(n)()
        return n.gradient

    eps = 1e-4
    hvp_fd = g(
        (
            -1.0 * first(g(z0 + 2 * eps * v))
            + 8.0 * first(g(z0 + eps * v))
            - 8.0 * first(g(z0 - eps * v))
            + first(g(z0 - 2 * eps * v))
        )
        * (1.0 / (12 * eps))
    )
    assert_field_close(hvp, hvp_fd, 1e-16, f"component.{op} HVP vs FD of 1st derivative")
