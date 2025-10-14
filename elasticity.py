import time
from math_utils import *
from sph_kernel import *

@ti.data_oriented
class Elasticity:
    def __init__(self, particle_system):

        self.ps = particle_system

        #TODO: allocate F, L for deformable particles only
        self.F = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.L = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)

        self.d2EdF2 = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.ps.particle_max_num, 4, 4))
        self.initialize()

    def initialize(self):

        print("[TODO]: initialize rest volume")
        print("[TODO]: initialize kernel correction L")

        pass

    @ti.kernel
    def compute_F(self):

        for p_i in ti.grouped(self.ps.x):
            pass

    @ti.kernel
    def compute_gradient_and_hessian(self, YM: float, PR: float):

        # stretch term, volume term
        for p_i in ti.grouped(self.ps.x):
            pass

    def PCG(self, x, b):

        pass

    @ti.kernel
    def compute_Ax(self, Ax: ti.template(), x: ti.template()):

        pass

    def solve(self, YM, PR, dt):

        print("Not implemented yet....")
        self.compute_F()
        self.compute_gradient_and_hessian(YM, PR)
        self.PCG(x=None, b=None)

        pass