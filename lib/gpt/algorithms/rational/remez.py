#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2026  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#                  2026  Christopher Kelly
#                  Adapted from the CPS library implementation
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

import gpt as g
import numpy as np
from . import rational_function

try:
    import mpmath as mp
    from mpmath import mpf

    HAVE_MPMATH = True
except ImportError:
    HAVE_MPMATH = False


class RemezApproximation:
    def __init__(self, func: callable, lo: float, hi: float, degree: int):
        """
        lo, hi: lower/upper bound on approximation range
        degree: the degree of the polynomial
        """
        self.func = func
        self.degree = degree
        self.param = None  # this will be an rrray of mpmath.mpf once Remez is called
        self.lo = lo
        self.hi = hi

    def approx(self, x):
        """
        The approximation, evaluated for a numerical or list-of-numerical values 'x'
        (Result will be in mpmath.mpf format)
        """

        if isinstance(x, (list, np.ndarray)):
            l = [self.approx(xx) for xx in x]
            if isinstance(x, list):
                return l
            else:
                return np.array(l)

        # Evaluate the rational form P(x)/Q(x) using coefficients from the solution vector param
        assert self.param is not None
        # Work backwards toward the constant term.
        n = self.degree
        d = self.degree
        yn = self.param[n]  # Highest order numerator coefficient
        for i in range(n - 1, -1, -1):
            yn = x * yn + self.param[i]

        yd = x + self.param[n + d]  # Highest degree coefficient = 1.0
        for i in range(n + d - 1, n, -1):
            yd = x * yd + self.param[i]

        return yn / yd


def remez(
    func: callable, lo: float, hi: float, degree: int, tolerance=1e-15, precision=50
) -> RemezApproximation:

    if not HAVE_MPMATH:
        raise Exception("Remez algorithm requires the mpmath module")

    mp.mp.prec = precision

    apstrt = lo
    apend = hi
    apwidt = apend - apstrt

    M_PI = mp.pi

    # CPS conventions used n and d to make numerator and denominator indexing more clear, even though they are the same
    n = degree
    d = degree
    neq = n + d + 1

    xx = [None for _ in range(n + d + 3)]
    mm = [None for _ in range(n + d + 2)]
    step = [None for _ in range(n + d + 2)]

    out = RemezApproximation(func, lo, hi, degree)

    delta = None
    spread = mpf(1e37)

    def initialGuess():
        # Supply initial guesses for solution points
        ncheb = neq
        # Degree of Chebyshev error estimate

        # Find ncheb+1 extrema of Chebyshev polynomial
        a = mpf(ncheb)
        mm[0] = mpf(apstrt)
        for i in range(1, ncheb):
            r = (1 - mp.cos(M_PI * i / a)) / 2
            r = (mp.exp(r) - 1) / mp.expm1(1)
            mm[i] = apstrt + r * apwidt

        mm[ncheb] = apend

        a *= 2
        for i in range(0, ncheb):
            r = (1 - mp.cos(M_PI * (2 * i + 1) / a)) / 2
            r = (mp.exp(r) - 1) / mp.expm1(1)
            xx[i] = apstrt + r * apwidt

    def stpini():
        # Initialize "step"
        xx[neq + 1] = mpf(apend)
        nonlocal delta
        delta = 0.25
        step[0] = xx[0] - apstrt
        for i in range(1, neq):
            step[i] = xx[i] - xx[i - 1]
        step[neq] = step[neq - 1]

    def equations():
        AA = mp.matrix(neq, neq)
        BB = mp.matrix(neq, 1)

        for i in range(neq):  # row
            x = xx[i]
            # the guess for this row
            y = out.func(x)
            # right-hand-side vector

            z = mpf(1)
            for j in range(n + 1):
                AA[i, j] = z
                z *= x

            z = mpf(1)
            for j in range(d):
                AA[i, n + 1 + j] = -y * z
                z *= x

            BB[i] = y * z
            # Right hand side vector

        # Solve AA XX = BB
        # param, resid = mp.qr_solve(AA,BB) #more accurate, but testing does not show it is needed
        # g.message("Equations resid=",resid)
        param = mp.lu_solve(AA, BB)
        assert param.rows == neq and param.cols == 1
        out.param = param

    def getErr(x):
        # Compute size and sign of the approximation error at x
        f = out.func(x)
        e = out.approx(x) - f
        sign = 1
        if f != mpf(0):
            e /= f
        if e < mpf(0):
            sign = -1
            e = -e
        return e, sign

    def search():
        # Search for error maxima and minima
        meq = neq + 1
        yy = [None for _ in range(meq)]

        nonlocal delta
        nonlocal spread

        eclose = mpf(1.0e30)
        farther = mpf(0)

        xx0 = apstrt

        # Search loop
        for i in range(meq):
            steps = 0
            xx1 = xx[i]
            if i == meq - 1:
                xx1 = apend
            xm = mm[i]
            ym, emsign = getErr(xm)
            q = step[i]
            xn = xm + q

            if xn < xx0 or xn >= xx1:  # Cannot skip over adjacent boundaries
                q = -q
                xn = xm
                yn = ym
                ensign = emsign
            else:
                yn, ensign = getErr(xn)
                if yn < ym:
                    q = -q
                    xn = xm
                    yn = ym
                    ensign = emsign

            while yn >= ym:  # March until error becomes smaller.
                steps += 1
                if steps > 10:
                    break
                ym = yn
                xm = xn
                emsign = ensign
                a = xm + q
                if a == xm or a <= xx0 or a >= xx1:
                    break  # Must not skip over the zeros either side.
                xn = a
                yn, ensign = getErr(xn)

            mm[i] = xm  # Position of maximum
            yy[i] = ym
            # Value of maximum

            if eclose > ym:
                eclose = ym
            if farther < ym:
                farther = ym

            xx0 = xx1  # Walk to next zero.
        # end of search loop

        q = farther - eclose  # Decrease step size if error spread increased
        if eclose != 0.0:
            q /= eclose  # Relative error spread
        if q >= spread:
            delta *= 0.5  # Spread is increasing; decrease step size
        spread = q
        # g.message("spread=", spread, " delta=",delta)

        for i in range(neq):
            q = yy[i + 1]
            if q != 0.0:
                q = yy[i] / q - mpf(1)
            else:
                q = mpf(0.0625)
            if q > mpf(0.25):
                q = mpf(0.25)
            q *= mm[i + 1] - mm[i]
            step[i] = q * delta

        step[neq] = step[neq - 1]

        for i in range(neq):  # Insert new locations for the zeros.
            xm = xx[i] - step[i]
            if xm <= apstrt:
                continue
            if xm >= apend:
                continue
            if xm <= mm[i]:
                xm = mpf(0.5) * (mm[i] + xx[i])
            if xm >= mm[i + 1]:
                xm = mpf(0.5) * (mm[i + 1] + xx[i])
            xx[i] = xm

    # The algorithm
    initialGuess()
    # g.message("Initial guess: ",xx)
    stpini()
    iter = 0
    while spread > tolerance:  # iterate until convergance

        if iter % 100 == 0:
            g.message(f"Iteration {iter}, spread {spread} delta {delta}")
        iter += 1
        equations()
        if delta < tolerance:
            raise Exception("Delta too small, try increasing precision")

        search()

    error, sign = getErr(mm[0])
    g.message(f"Converged at {iter} iterations, error = {error}")
    return RemezPFE(out, precision)


class RemezPartialFractionExpansion:
    def __init__(self):
        self.norm = None
        self.poles = None
        self.zeros = None

    def __str__(self):
        return f"remez({self.norm}, {len(self.poles)}, {len(self.zeros)})"


def RemezPFE(approx: RemezApproximation, precision=100) -> RemezPartialFractionExpansion:
    """Compute the partial-fraction expansion of a Remez rational approximation"""

    n = approx.degree
    d = approx.degree
    roots = [None for _ in range(n)]
    poles = [None for _ in range(d)]
    mp.mp.prec = precision
    norm = None

    def polyEval(x, poly, size):
        # Evaluate the polynomial
        f = poly[size]
        for i in range(size - 1, -1, -1):
            f = f * x + poly[i]
        return f

    def polyDiff(x, poly, size):
        # Evaluate the differential of the polynomial
        df = size * poly[size]
        for i in range(size - 1, 0, -1):  # yes this is i>0 not i>=0
            df = df * x + i * poly[i]
        return df

    def rtnewt(poly, i, x1, x2, xacc):
        # Newton's method to calculate roots
        JMAX = 1000
        rtn = 0.5 * (x1 + x2)
        for j in range(1, JMAX + 1):
            f = polyEval(rtn, poly, i)
            df = polyDiff(rtn, poly, i)
            dx = f / df
            # g.message(j, " ", dx)
            rtn -= dx
            if (x1 - rtn) * (rtn - x2) < mpf(0.0):
                g.message("Warning: Jumped out of brackets in rtnewt")
            if mp.fabs(dx) < xacc:
                return rtn
        g.message("Warning: Maximum number of iterations exceeded in rtnewt")
        return 0

    def root():
        # Calculate the roots of the approximation
        upper = mpf(1)
        lower = mpf(-100000)
        tol = mpf(1e-20)

        neq = n + d + 1
        poly = [None for _ in range(neq + 1)]

        # First find the numerator roots
        for i in range(n + 1):
            poly[i] = approx.param[i]
        for i in range(n - 1, -1, -1):
            roots[i] = rtnewt(poly, i + 1, lower, upper, tol)
            if roots[i] == mpf(0.0):
                raise Exception("Failure to converge on root")

            poly[0] = -poly[0] / roots[i]
            for j in range(1, i + 1):
                poly[j] = (poly[j - 1] - poly[j]) / roots[i]

        # Now find the denominator roots
        poly[d] = mpf(1)
        for i in range(d):
            poly[i] = approx.param[n + 1 + i]
        for i in range(d - 1, -1, -1):
            poles[i] = rtnewt(poly, i + 1, lower, upper, tol)
            if poles[i] == mpf(0.0):
                raise Exception("Failure to converge on root")

            poly[0] = -poly[0] / poles[i]
            for j in range(1, i + 1):
                poly[j] = (poly[j - 1] - poly[j]) / poles[i]

        nonlocal norm
        norm = approx.param[n]
        g.message("Normalisation constant is ", norm)
        for i in range(n):
            g.message(i, " root = ", roots[i])
        for i in range(d):
            g.message(i, " pole = ", poles[i])

    def pfe(res, poles, norm):
        # Evaluate the partial fraction expansion of the rational function with res roots and poles poles.  Result is overwritten on input arrays.
        numerator = [None for _ in range(n)]
        denominator = [None for _ in range(n)]

        # Construct the polynomials explicitly
        for i in range(1, n):
            numerator[i] = mpf(0)
            denominator[i] = mpf(0)

        numerator[0] = mpf(1)
        denominator[0] = mpf(1)

        for j in range(n):
            for i in range(n - 1, -1, -1):
                numerator[i] *= -res[j]
                denominator[i] *= -poles[j]
                if i > 0:
                    numerator[i] += numerator[i - 1]
                    denominator[i] += denominator[i - 1]
        # Convert to proper fraction form.
        # Fraction is now in the form 1 + n/d, where O(n)+1=O(d)
        for i in range(n):
            numerator[i] -= denominator[i]

        poles = sorted(poles)

    # The algorithm
    root()

    out = RemezPartialFractionExpansion()
    # PFE
    r = [roots[i] for i in range(n)]
    p = [poles[i] for i in range(n)]
    pfe(r, p, norm)

    # convert to float as mpmath no longer needed
    out.norm = float(norm)
    out.poles = [float(pp) for pp in p]
    out.zeros = [float(rr) for rr in roots]
    return out
