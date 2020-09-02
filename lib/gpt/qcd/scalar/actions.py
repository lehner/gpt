import gpt


class phi4:
    def __init__(self, phi, m, l):
        self.phi = phi
        self.grid = phi.grid
        self.Nd = self.grid.nd
        self.m = m
        self.l = l
        self.kappa = (1 - 2.0 * l) / (2 * self.Nd + m ** 2)
        self.J = gpt.lattice(self.phi)

    def __call__(self):
        self.J[:] = 0
        for mu in range(self.Nd):
            self.J += gpt.cshift(self.phi, mu, 1)
        act = -2.0 * self.kappa * gpt.inner_product(self.J, gpt.adj(self.phi)).real

        p2 = gpt.norm2(self.phi)
        act += p2 + self.l * (p2 - 1.0) ** 2

        return act

    def pre_force(self):
        self.J[:] = 0
        for mu in range(self.Nd):
            self.J += gpt.cshift(self.phi, mu, 1)
            self.J += gpt.cshift(self.phi, mu, -1)

    def force(self, mu):
        frc = gpt.lattice(self.phi)
        frc @= -2.0 * self.kappa * self.J
        frc += 2.0 * self.phi
        if self.l != 0.0:
            frc += 4.0 * self.l * gpt.adj(self.phi) * self.phi * self.phi
            frc += 4.0 * self.l * self.phi
        frc[:].imag = 0
        return frc
