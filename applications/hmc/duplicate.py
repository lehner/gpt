#!/usr/bin/env python3
import gpt as g

source = g.default.get("--source", None)
destination = g.default.get("--destination", None)
n = g.default.get_ivec("--n", None, 4)

U = g.load(source)
Lold = U[0].grid.gdimensions
Lnew = [n[i] * Lold[i] for i in range(len(n))]
grid = g.grid(Lnew, g.double)
Unew = []
for u in U:
    unew = g.mcolor(grid)
    for n0 in range(n[0]):
        for n1 in range(n[1]):
            for n2 in range(n[2]):
                for n3 in range(n[3]):
                    g.message(n0, n1, n2, n3)
                    if grid.processor == 0:
                        unew[
                            n0*Lold[0]:(n0+1)*Lold[0],
                            n1*Lold[1]:(n1+1)*Lold[1],
                            n2*Lold[2]:(n2+1)*Lold[2],
                            n3*Lold[3]:(n3+1)*Lold[3]
                        ] = u[
                            0:Lold[0],
                            0:Lold[1],
                            0:Lold[2],
                            0:Lold[3]
                        ]
                    else:
                        unew[
                            0:0,
                            0:0,
                            0:0,
                            0:0
                        ] = u[
                            0:0,
                            0:0,
                            0:0,
                            0:0
                        ]
    Unew.append(unew)

g.save(destination, Unew, g.format.nersc())
