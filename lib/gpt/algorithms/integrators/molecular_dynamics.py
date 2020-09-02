import gpt


class integrator:
    def __init__(self, N, i0, i1):
        self.N = N
        self.i0 = i0
        self.i1 = i1

    def get_act(self):
        return self.i0.get_act() + self.i1.get_act()


class leap_frog(integrator):
    def __call__(self, tau):
        eps = tau / self.N

        dt0 = dt1 = 0.0

        dt0 += self.i0(eps * 0.5)
        for i in range(self.N):
            dt1 += self.i1(eps)
            if i != self.N - 1:
                dt0 += self.i0(eps)
        dt0 += self.i0(eps * 0.5)

        gpt.message(f"LeapFrog Timings = dt0 {dt0:g} secs, dt1 {dt1:g} secs")
