#!/usr/bin/env python3
import gpt as g
#grid = g.grid([64,64,64,64], g.double)
grid = g.grid([32,32,32,32], g.double)
#grid = g.grid([32,16,16,16], g.double)
#grid = g.grid([16,16,16,32], g.double)
#grid = g.grid([2*4,4*3,3*4,3*3*4], g.double)
m1 = g.mcolor(grid)
m2 = g.mcolor(grid)
m3 = g.mcolor(grid)
rng = g.random("D")
rng.cnormal([m1,m2,m3])
m3ref = g(m1*m2)
code = []
ti = g.stencil.tensor_instructions

for i in range(3):
    for j in range(3):
        for l in range(3):
            dst = 3*i + j
            code.append(
                (0,dst,ti.mov if l == 0 else ti.inc,1.0,[(1,0,3*i + l),(2,0,3*l + j)])
            )

segments = [(3, 9)]
ein = g.stencil.tensor(m1, [(0, 0, 0, 0), (1, 0, 0, 0)], code, segments)

ein(m3,m1,m2)
g.message(g.norm2(m3 - m3ref))


for osites_per_instruction in [1,4,8,16,32,64]:
    for osites_per_cache_block in [2048*4, 4096*4, 8192*4]:
        ein.memory_access_pattern(osites_per_instruction, osites_per_cache_block)

        g.message(osites_per_instruction, osites_per_cache_block)
        t=g.timer("d")
        t("expr")
        for i in range(300):
            g.eval(m3,m1*m2)
        t("stencil")
        for i in range(300):
            ein(m3,m1,m2)
        t()
        g.message(t)
        eps2 = g.norm2(m3 - m3ref) / g.norm2(m3)
        assert eps2 < 1e-25
        g.message(eps2)


# D_{a2,a1} = epsilon_{a1,b1,c1}*epsilon_{a2,b2,c2}*spin_transpose(Q1_{b1,b2})*Q2_{c1,c2}
Q1 = g.mspincolor(grid)
Q2 = g.mspincolor(grid)
rng.cnormal([Q1,Q2])
eps = g.epsilon(Q1.otype.shape[2])
code = []
acc = {}
for i in range(4):
    for j in range(4):
        for l in range(4):
            for i1, sign1 in eps:
                for i2, sign2 in eps:
                    dst = (i*4 + j)*9 + i2[0]*3 + i1[0]
                    aa = (4*i + l)*9 + i1[1]*3 + i2[1]
                    bb = (4*j + l)*9 + i1[2]*3 + i2[2]
                    if dst not in acc:
                        acc[dst] = True
                        mode = ti.mov if sign1 * sign2 > 0 else ti.mov_neg
                    else:
                        mode = ti.inc if sign1 * sign2 > 0 else ti.dec
                    assert dst >= 0 and dst < 12*12
                    assert aa >= 0 and aa < 12*12
                    assert bb >= 0 and bb < 12*12
                    code.append(
                        (0,dst,mode,1.0,[(1,0,aa),(2,0,bb)])
                    )

g.message(len(code))
segments = [(len(code) // 16, 16)]
ein = g.stencil.tensor(Q1, [(0, 0, 0, 0), (1, 0, 0, 0)], code, segments)

R = g.mspincolor(grid)
R[:] = 0
ein(R, Q1, Q2)

R2 = g.qcd.baryon.diquark(Q1,Q2)

g.message(g.norm2(R - R2) / g.norm2(R))
#
#            D[i2[0], i1[0]] += sign1 * sign2 * Q1[i1[1], i2[1]] * g.transpose(Q2[i1[2], i2[2]])
for osites_per_instruction in [1,4,8,16,32,64]:
    for osites_per_cache_block in [2048*4, 4096*4, 8192*4]:
        ein.memory_access_pattern(osites_per_instruction, osites_per_cache_block)

        g.message(osites_per_instruction, osites_per_cache_block)
        t=g.timer("d")
        t("diquark")
        for i in range(30):
            g.qcd.baryon.diquark(Q1,Q2)
        t("stencil")
        for i in range(30):
            ein(R, Q1, Q2)
        t()
        g.message(t)
        g.message(g.norm2(R - R2) / g.norm2(R))    
