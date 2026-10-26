#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2022  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
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
#    Reference: https://arxiv.org/pdf/1412.6980.pdf
#
import gpt as g
import numpy as np
from gpt.algorithms.optimize.optimizer import optimizer


def new(a):
    if g.util.is_num(a):
        return a
    elif isinstance(a, np.ndarray):
        return np.copy(a)
    return a.new()


def nfloats(a):
    if g.util.is_num(a):
        # treat as complex
        return 2
    elif isinstance(a, np.ndarray):
        return a.size * 2
    return a.nfloats()


def set_element(a, i, b):
    # fields and arrays are updated in place (others may hold them, e.g. the
    # functions of g.ml and the composites that share their parameters);
    # numbers are replaced
    if isinstance(a[i], (g.lattice, g.tensor)):
        if g.util.is_num(b):
            a[i][:] = b
        else:
            a[i] @= b
    elif isinstance(a[i], np.ndarray):
        if not np.iscomplexobj(a[i]) and np.iscomplexobj(b):
            # real parameters: the updates are complex with vanishing imaginary part
            assert not np.any(np.abs(np.imag(b)) > 0)
            b = np.real(b)
        a[i][...] = b
    else:
        a[i] = b


def set_value(a, b):
    for i in range(len(a)):
        set_element(a, i, b)


def _sums(a, square):
    # per parameter (a number, a component of a numpy array, or a whole lattice): the sum of |a|^2 over its components
    # (square) or of re + im (the moments v and variances, which hold the real and imaginary parts' squares)
    if g.util.is_num(a):
        a = complex(a)
        return [abs(a) ** 2 if square else a.real + a.imag], [1]
    if isinstance(a, np.ndarray):
        x = np.asarray(a).reshape(-1)
        return list(np.abs(x) ** 2 if square else x.real + np.imag(x)), [1] * len(x)
    if square:
        return [g.norm2(a)], [nfloats(a) // 2]
    # (re, im >= 0: their sum is the norm2 of sqrt(re) + i sqrt(im))
    h = g(g.component.sqrt(g.component.real(a)) + 1j * g.component.sqrt(g.component.imag(a)))
    return [g.norm2(h)], [nfloats(a) // 2]


class report:
    """The state of an Adam run in a readable form (str) and as a dict (as_dict).  Per parameter (a number, a component of
    a numpy array, or a whole lattice field):

      r = |mhat|^2 / vhat  (bias-corrected moments; summed over the components of a field)

    the squared mean of the recent gradients (window ~ 1 / (1 - beta1) steps) over their mean square (window
    ~ 1 / (1 - beta2) steps): r = 1 for the same gradient in every step, r ~ r0 for gradients that are pure noise
    around zero, with r0 = sum_i w_i^2 (w the weights of mhat; 0.053 for beta1 = 0.9 and many steps).  Adam moves a
    parameter by alpha sqrt(r) per step (for sqrt(vhat) above the regulator).  Classes:

      dead    sqrt(vhat) below eps_regulator per component: no gradient, the regulator sets the step
      drift   r >= 0.5: a consistent gradient, steps of ~alpha in one direction (far from the optimum, or alpha small)
      noise   r <= 2 r0: the gradient is noise around zero (at the optimum, too noisy to resolve it, or steps so large
              that the parameter oscillates; batch_var distinguishes)
      mixed   in between
      ?       not classified: r0 >= 0.25 (beta1 below ~0.8 or t < 4), too close to 0.5 to tell drift from noise

    With batch_var (per parameter the variance of one step's gradient estimate, e.g. the batch variance / batch size,
    averaged over the steps; the same structure as the gradient), per group:

      batch share   sum batch_var / sum (vhat - |mhat|^2): the part of the step-to-step variation of the gradient
                    that the batch noise explains.  ~1: a larger batch reduces it; << 1: the variation comes from
                    elsewhere (different data in different steps, drift), which more samples of the same kind per
                    step do not reduce
      noise/signal  sum batch_var / sum max(|mhat|^2 - r0 (vhat - |mhat|^2), 0): the batch noise over the squared
                    mean gradient; the batch would have to grow by this factor for the signal to match the noise
    """

    def __init__(self, opt, state, groups, names, batch_var, detail):
        self.alpha, self.beta1, self.beta2, self.t = (
            opt.alpha,
            opt.beta1,
            opt.beta2,
            state.t,
        )
        if state.m is None or state.t == 0:
            self.params, self.groups = None, {}
            return
        b1, b2, t = opt.beta1, opt.beta2, state.t
        self.r0 = (1 - b1) ** 2 * (1 - b1 ** (2 * t)) / ((1 - b1**2) * (1 - b1**t) ** 2)
        M2, V, N, BV = [], [], [], []
        for nu in range(len(state.m)):
            m2, n = _sums(state.m[nu], True)
            v, _ = _sums(state.v[nu], False)
            M2 += m2
            V += v
            N += n
            if batch_var is not None:
                BV += _sums(batch_var[nu], False)[0]
        M2 = np.array(M2) / (1 - b1**t) ** 2
        V = np.array(V) / (1 - b2**t)
        N = np.array(N, dtype=float)
        n = len(M2)
        groups = ["all"] * n if groups is None else list(groups)
        names = [str(i) for i in range(n)] if names is None else list(names)
        assert (
            len(groups) == n and len(names) == n
        ), f"report: {n} parameters, {len(groups)} groups, {len(names)} names"
        dead = V <= N * opt.eps_regulator**2
        r = np.where(dead, 0.0, M2 / np.where(dead, 1.0, V))
        self.reliable = self.r0 < 0.25
        cls = np.where(
            dead,
            "dead",
            np.where(r >= 0.5, "drift", np.where(r <= 2 * self.r0, "noise", "mixed")),
        )
        cls = np.where(dead | self.reliable, cls, "?")
        S2 = np.maximum(V - M2, 0.0)
        BV = np.array(BV) if batch_var is not None else None
        self.params = dict(name=names, group=groups, sqrt_r=np.sqrt(r).tolist(), cls=cls.tolist())
        self.detail = detail
        self.groups = {}
        for gr in dict.fromkeys(groups):
            idx = np.array([i for i in range(n) if groups[i] == gr])
            live = idx[~dead[idx]]
            sr = np.sqrt(r[live])
            d = dict(
                n=len(idx),
                **{c: int(np.sum(cls[idx] == c)) for c in ["drift", "mixed", "noise", "dead"]},
                sqrt_r=(
                    [float(np.min(sr)), float(np.median(sr)), float(np.max(sr))]
                    if len(live)
                    else None
                ),
                step=float(self.alpha * np.median(sr)) if len(live) else 0.0,
                index=idx.tolist(),
            )
            if BV is not None and len(live):
                s2, bv = np.sum(S2[live]), np.sum(BV[live])
                sig = np.sum(np.maximum(M2[live] - self.r0 * S2[live], 0.0))
                d["batch_share"] = float(bv / s2) if s2 > 0 else float("nan")
                d["noise_signal"] = float(bv / sig) if sig > 0 else float("inf")
            d["verdict"] = self._verdict(d)
            self.groups[gr] = d

    def _verdict(self, d):
        out = []
        n = d["n"]
        if d["dead"]:
            out.append(f"{d['dead']} dead (no gradient)")
        if not self.reliable:
            return out
        live = n - d["dead"]
        if live and d["drift"] >= live / 2:
            out.append(
                "drifting: steps of ~alpha in a consistent direction (far from the optimum or alpha small; if the "
                "cost gets worse meanwhile, the steps are too large)"
            )
        elif live and d["noise"] >= live / 2:
            out.append("noise-dominated: the parameters jitter around their current values")
        sh = d.get("batch_share")
        if sh is not None and live and d["drift"] < live / 2:
            if sh >= 0.5:
                out.append(
                    f"the batch noise accounts for most of the step-to-step variation (share {sh:.2f}): a larger "
                    "batch helps"
                )
            elif sh < 0.2:
                out.append(
                    f"the batch noise accounts for little of the step-to-step variation (share {sh:.2f}): it comes from the "
                    "data changing between steps or from drift; more independent data per step helps, more samples "
                    "of the same data do not"
                )
        return out

    def as_dict(self):
        return dict(
            t=self.t,
            alpha=self.alpha,
            beta1=self.beta1,
            beta2=self.beta2,
            r0=getattr(self, "r0", None),
            groups=self.groups,
            params=self.params,
        )

    def __str__(self):
        if self.params is None:
            return "adam report: no steps yet"
        hdr = (
            f"adam report: t {self.t}, alpha {self.alpha}, beta1 {self.beta1}, beta2 {self.beta2}; sqrt(r) = |mhat| / "
            f"sqrt(vhat): 1 the same gradient every step, {np.sqrt(self.r0):.2f} pure noise"
        )
        lines = [hdr]
        if not self.reliable:
            lines.append(
                f"  (r0 = {self.r0:.2f}: too few steps or beta1 too small to tell drift from noise)"
            )
        w = max(12, max(len(k) for k in self.groups))
        has_b = any("batch_share" in d for d in self.groups.values())
        lines.append(
            f"  {'group':{w}s} {'n':>5s}  sqrt(r) min/median/max   step  drift mixed noise  dead"
            + ("  batch share  noise/signal" if has_b else "")
        )
        for k, d in self.groups.items():
            q = " ".join(f"{x:5.2f}" for x in d["sqrt_r"]) if d["sqrt_r"] else "    -     -     -"
            line = (
                f"  {k:{w}s} {d['n']:5d}  {q:22s}  {d['step']:.1e} {d['drift']:5d} {d['mixed']:5d} {d['noise']:5d} "
                f"{d['dead']:5d}"
            )
            if "batch_share" in d:
                line += f"  {d['batch_share']:11.2f}  {d['noise_signal']:12.3g}"
            lines.append(line)
        for k, d in self.groups.items():
            for v in d["verdict"]:
                lines.append(f"  {k}: {v}")
            if self.detail and d["n"] <= self.detail:
                lines.append(
                    f"  {k}: "
                    + ", ".join(
                        f"{self.params['name'][i]} {self.params['sqrt_r'][i]:.2f}"
                        f"{'' if self.params['cls'][i] in ('mixed',) else ' ' + self.params['cls'][i]}"
                        for i in d["index"]
                    )
                )
        return "\n".join(lines)


class adam(optimizer):
    @g.params_convention(
        eps=1e-8,
        maxiter=1000,
        alpha=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps_regulator=1e-8,
        log_functional_every=10,
    )
    def __init__(self, params):
        super().__init__()
        self.eps = params["eps"]
        self.maxiter = params["maxiter"]
        self.alpha = params["alpha"]
        self.beta1 = params["beta1"]
        self.beta2 = params["beta2"]
        self.eps_regulator = params["eps_regulator"]
        self.nf = params["log_functional_every"]

    def new_state(self):
        # the moments and the step count (kept across the calls of a run)
        class context:
            m = None
            t = 0

        return context

    def report(self, state, groups=None, names=None, batch_var=None, detail=16):
        # run.report(...): the diagnostics of the moments (see report); groups and names: one per parameter (numbers,
        # numpy components, lattice fields in the order of the optimized fields); detail: list the parameters of groups
        # up to this size
        return report(self, state, groups, names, batch_var, detail)

    def iterate(self, f, x, dx_indices, context, t):

        for i in range(self.maxiter):
            context.t += 1

            # x = theta(t-1)
            dx = [x[i] for i in dx_indices]
            gt = f.gradient(x, dx)
            gt2 = []

            # momentum vectors
            if context.m is None:
                context.m = [new(a) for a in gt]
                context.v = [new(a) for a in gt]
                context.mhat = [new(a) for a in gt]
                context.vhat = [new(a) for a in gt]
                context.epsfield = [new(a) for a in gt]

                set_value(context.m, 0)
                set_value(context.v, 0)
                set_value(context.epsfield, self.eps_regulator)
            else:
                # (numbers may change type, e.g. complex -> numpy.complex128)
                m0, g0 = context.m[0], gt[0]
                assert (g.util.is_num(m0) and g.util.is_num(g0)) or type(m0) is type(g0)

            for a in gt:
                ar = g(g.component.real(a))
                ai = g(g.component.imag(a))
                gt2.append(g(g.component.multiply(ar, ar) + 1j * g.component.multiply(ai, ai)))

            for nu in range(len(dx)):
                set_element(context.m, nu, self.beta1 * context.m[nu] + (1 - self.beta1) * gt[nu])
                set_element(context.v, nu, self.beta2 * context.v[nu] + (1 - self.beta2) * gt2[nu])

                set_element(context.mhat, nu, (1.0 / (1.0 - self.beta1**context.t)) * context.m[nu])
                set_element(context.vhat, nu, (1.0 / (1.0 - self.beta2**context.t)) * context.v[nu])

                vhat_nu_real = g(g.component.sqrt(g.component.real(context.vhat[nu])))
                vhat_nu_imag = g(g.component.sqrt(g.component.imag(context.vhat[nu])))

                reg_mhat_real = g.component.multiply(
                    g.component.real(context.mhat[nu]),
                    g.component.inv(context.epsfield[nu] + vhat_nu_real),
                )
                reg_mhat_imag = g.component.multiply(
                    g.component.imag(context.mhat[nu]),
                    g.component.inv(context.epsfield[nu] + vhat_nu_imag),
                )

                # make sure object type is correct
                tmp = [new(gt[nu])]
                set_element(tmp, 0, -self.alpha * (reg_mhat_real + 1j * reg_mhat_imag))
                set_element(x, dx_indices[nu], g.group.compose(tmp[0], x[dx_indices[nu]]))

            rs = (sum([g.norm2(x) for x in gt]) / sum([nfloats(s) for s in gt])) ** 0.5

            self.log_convergence(i, rs, self.eps)

            if i % self.nf == 0:
                self.log(
                    f"iteration {i}: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}, alpha = {self.alpha}, beta1 = {self.beta1}, beta2 = {self.beta2}"
                )

            if rs <= self.eps:
                self.log(
                    f"converged in {i + 1} iterations: f(x) = {f(x):.15e}, |df|/sqrt(dof) = {rs:e}"
                )
                return True

        if self.maxiter > 1:
            self.log(
                f"NOT converged in {i + 1} iterations;  |df|/sqrt(dof) = {rs:e} / {self.eps:e}"
            )
        return False
