import gpt


def inv_ne(matrix, inverter):

    otype = matrix.otype

    tmp = gpt.lattice(matrix.F_grid, otype)

    def inv(dst_sc, src_sc):

        matrix.N.adj()(tmp, src_sc)
        inverter(matrix.NDagN)(dst_sc, tmp)

    m = gpt.matrix_operator(
        mat=inv,
        inv_mat=matrix.op.M,
        adj_inv_mat=matrix.op.M.adj(),
        adj_mat=None,  # implement adj_mat when needed
        otype=otype,
        zero=(True, False),
        grid=matrix.F_grid,
        cb=None,
    )

    m.ImportPhysicalFermionSource = matrix.ImportPhysicalFermionSource
    m.ExportPhysicalFermionSolution = matrix.ExportPhysicalFermionSolution

    return m
