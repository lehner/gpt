import gpt as g
import numpy as np

def test_factor_unary():
    grid = g.grid([8, 8, 8, 16], g.double)
    v = g.vspincolor(grid)
    adj_v = g.adj(v)
    result = g(adj_v + v)
    assert result.otype.__name__ == "ot_vector_spin_color(4,3)"

def test_addition_with_complex():
    grid = g.grid([8, 8, 8, 16], g.double)
    singlet = g.singlet(grid)
    result = g(singlet + 2.0j)
    assert result.otype.__name__ == "ot_singlet"

def test_automatic_embedding():
    grid = g.grid([8, 8, 8, 16], g.double)
    singlet = g.singlet(grid)
    result = g(singlet + 2.0j)
    assert result.otype.__name__ == "ot_singlet"

def test_explicit_casting():
    grid = g.grid([8, 8, 8, 16], g.double)
    singlet = g.singlet(grid)
    casted = g.convert(singlet, g.ot_singlet())
    assert casted.otype.__name__ == "ot_singlet"

if __name__ == "__main__":
    test_factor_unary()
    test_addition_with_complex()
    test_automatic_embedding()
    test_explicit_casting()
    print("All tests passed.")
