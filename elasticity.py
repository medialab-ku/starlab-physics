import time
from math_utils import *
from sph_kernel import *

@ti.data_oriented
class Elasticity:
    def __init__(self, particle_system):

        self.ps = particle_system

        #TODO: allocate F, L for deformable particles only
        self.L = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.F = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.P = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)

        self.grad  = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.x_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.d2EdF2 = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.ps.particle_max_num, 4, 4))
        self.gradW = spiky_kernel_derivative

        self.initialize()

    @ti.kernel
    def initialize(self):

        # set initial neighbours
        for p_i in ti.grouped(self.ps.x):
            p_i0 = self.ps.cur2ori[p_i]
            self.ps.solid_neighbors_num[p_i0] = self.ps.particle_neighbors_num[p_i]
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                self.ps.solid_neighbors[p_i0, j] = self.ps.particle_neighbors[p_i, j]

        #TODO: set rest volume
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

        # compute L
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            Dm = ti.math.mat3(0.0)
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xij0 = self.ps.x0[p_i0] - self.ps.x0[p_j0]
                Dm -= self.ps.m_V0[p_j] * self.gradW(xij0, self.ps.support_radius).outer_product(xij0)

            self.L[p_i] = Dm.inverse()

    @ti.kernel
    def compute_F(self, x: ti.template()):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            Ds_i = ti.math.mat3(0.0)
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xji = x[p_j] - x[p_i]
                xij0 = self.ps.x0[p_i0] - self.ps.x0[p_j0]
                Ds_i += self.ps.m_V0[p_j] * xji.outer_product(self.gradW(xij0, self.ps.support_radius))

            self.F[p_i] = Ds_i @ self.L[p_i].transpose()

    @ti.func
    def ssvd(self, F):
        U, sig, V = ti.svd(F)
        if U.determinant() < 0:
            for i in ti.static(range(3)): U[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        if V.determinant() < 0:
            for i in ti.static(range(3)): V[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        return U, sig, V

    @ti.kernel
    def compute_P(self, YM: float, PR: float):

        # stretch term, volume term
        mu = YM / 2.0 * (1.0 + PR)


        print("TODO: volume expansion")

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            F_i = self.F[p_i]
            U, sig, V = self.ssvd(F_i)
            R_i = U @ V.transpose()

            self.P[p_i] = 2 * mu * (F_i - R_i)

    @ti.kernel
    def compute_gradient(self, dt: float):

        # dtSq = dt ** 2
        for p_i in ti.grouped(self.ps.x):

            self.grad[p_i] = ti.math.vec3(0.0)
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            PL_i = self.P[p_i] @ self.L[p_i]
            f_s = ti.math.vec3(0.0)

            p_i0 = self.ps.cur2ori[p_i]
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]

                xij0 = self.ps.x0[p_i0] - self.ps.x0[p_j0]
                PL_j = self.P[p_j] @ self.L[p_j]
                f_s += self.ps.m_V0[p_j] * (PL_i - PL_j) * self.gradW(xij0, self.ps.support_radius)

            self.grad[p_i] = self.ps.m_V0[p_i] * dt * f_s

    def PCG(self, x, b):

        pass

    @ti.kernel
    def compute_Ax(self, Ax: ti.template(), x: ti.template()):

        pass

    def solve(self, YM, PR, dt):

        print("Not implemented yet....")

        add(self.x_tmp, self.ps.x, dt, self.ps.v)
        self.compute_F(self.x_tmp)
        self.compute_P(YM, PR)
        self.compute_gradient(dt)

        # TODO: v += dt * M^-1 * (-grad)
        # add(self.x_tmp, self.ps.x, dt, self.ps.v)
        # self.PCG(x=None, b=None)



        pass