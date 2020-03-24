#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt as g

class power_iteration:

    def __init__(self, params):
        self.params = params
        self.tol = params["eps"]
        self.maxit = params["maxiter"]

    def __call__(self,mat,src):
        verbose="power_iteration" in g.default.verbose

        dst,tmp=g.lattice(src),g.copy(src)

        tmp /= g.norm2(tmp)**0.5

        ev_prev=None
        for it in range(self.maxit):
            mat(tmp,dst)
            ev=g.norm2(dst)**0.5
            if verbose:
                g.message("eval_max[ %d ] = %g" % (it,ev))
            tmp @= (1.0/ev) * dst
            if not ev_prev is None:
                if abs(ev - ev_prev) < self.tol*ev:
                    if verbose:
                        g.message("Converged")
                    return (ev,tmp,True)
            ev_prev=ev

        return (ev,tmp,False)
