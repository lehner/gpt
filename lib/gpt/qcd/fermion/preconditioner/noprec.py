import gpt

class noprec:
    def __init__(self, op):
        self.op = op
        self.otype = op.otype
        self.F_grid_eo = op.F_grid_eo
        self.F_grid = op.F_grid
        self.U_grid = op.U_grid
        self.tmp = gpt.lattice(self.F_grid, self.otype)
        self.ImportPhysicalFermionSource = self.op.ImportPhysicalFermionSource
        self.ExportPhysicalFermionSolution = self.op.ExportPhysicalFermionSolution
        self.Dminus = self.op.Dminus
        self.ExportPhysicalFermionSource = self.op.ExportPhysicalFermionSource

        def _N(op, ip):
            self.op.M.mat(op, ip)

        def _NDag(op, ip):
            self.op.M.adj().mat(op, ip)

        def _NDagN(op, ip):
            _N(self.tmp, ip)
            _NDag(op, self.tmp)

        self.N = gpt.matrix_operator(
            mat=_N, adj_mat=_NDag, otype=op.otype, grid=self.F_grid
        )
        self.NDagN = gpt.matrix_operator(
            mat=_NDagN,
            adj_mat=_NDagN,
            otype=op.otype,
            grid=self.F_grid
        )
