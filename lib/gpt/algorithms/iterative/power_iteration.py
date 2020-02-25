#
# GPT
#
# Authors: Christoph Lehner 2020
#
import gpt as g

def power_iteration(mat, src,tol,maxit,verbose=True):
    dst,tmp=g.lattice(src),g.copy(src)

    tmp = g.eval( (1.0/g.norm2(tmp)**0.5) * tmp )

    ev_prev=None
    for it in range(maxit):
        mat(tmp,dst)
        ev=g.norm2(dst)**0.5
        if verbose:
            g.message("Iteration %d %g" % (it,ev))
        tmp=g.eval( (1.0/ev) * dst )
        if not ev_prev is None:
            if abs(ev - ev_prev) < tol*ev:
                if verbose:
                    g.message("Converged")
                return (ev,tmp,True)
        ev_prev=ev

    return (ev,tmp,False)
