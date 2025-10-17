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
        self.P_s = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.P_v = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.K = ti.Vector.field(n=self.ps.cache_size, dtype=float, shape=self.ps.particle_max_num)
        self.LgradW = ti.Matrix.field(3, self.ps.cache_size, dtype=float, shape=self.ps.particle_max_num)

        self.ZE = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.grad  = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.x_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.a = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.d2EdF2 = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.ps.particle_max_num, 4, 4))
        self.gradW = spiky_kernel_derivative
        self.W = cubic_kernel



    @ti.kernel
    def initialize(self):
        eps = 1e-12
        # set initial neighbours
        for p_i in ti.grouped(self.ps.x):
            p_i0 = self.ps.cur2ori[p_i]
            self.ps.solid_neighbors_num[p_i0] = 0
            
            for j in range(self.ps.particle_neighbors_num[p_i]):
                p_j = self.ps.particle_neighbors[p_i, j]
                if self.ps.material[p_j] != self.ps.material_solid:
                    continue

                p_j0 = self.ps.cur2ori[p_j]
                if self.ps.solid_neighbors_num[p_i0] < self.ps.cache_size:
                    self.ps.solid_neighbors[p_i0, self.ps.solid_neighbors_num[p_i0]] = p_j0
                    self.ps.solid_neighbors_num[p_i0] += 1

            # print(self.ps.solid_neighbors_num[p_i0])

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue
        
            p_i0 = self.ps.cur2ori[p_i]
            sum_Wij0 = self.W(0.0, self.ps.support_radius)
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xji0 = self.ps.x0[p_j] - self.ps.x0[p_i]
                sum_Wij0 += self.W(xji0.norm(), self.ps.support_radius)
        
            self.ps.m_V0[p_i] = 1.0 / sum_Wij0
            self.ps.m[p_i] = self.ps.density0[p_i] * self.ps.m_V0[p_i]
            self.ps.m_inv[p_i] = 1.0 / self.ps.m[p_i]
                
        # compute L
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            Dm = ti.math.mat3(0.0)
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xji0 = self.ps.x0[p_j] - self.ps.x0[p_i]
                # Dm += xji0.outer_product(xji0)
                Dm -= self.ps.m_V0[p_j] * self.gradW(xji0, self.ps.support_radius).outer_product(xji0)
            
            # print(f"Dm: {ti.math.determinant(Dm)}")
            self.L[p_i0] = Dm.inverse()
            # self.F[p_i] = Dm @ self.L[p_i0].transpose()
            # print(f"F: {self.F[p_i]}")

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            Li   = self.L[p_i0]

            # compute L @ gradW
            for k in range(self.ps.solid_neighbors_num[p_i0]):
                p_k0 = self.ps.solid_neighbors[p_i0, k]
                p_k  = self.ps.ori2cur[p_k0]
                xik0  = self.ps.x0[p_i] - self.ps.x0[p_k]
                gradWik = Li @ self.gradW(xik0, self.ps.support_radius)
                self.LgradW[p_i0][0, k] = gradWik[0]
                self.LgradW[p_i0][1, k] = gradWik[1]
                self.LgradW[p_i0][2, k] = gradWik[2]

            # computeK
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j  = self.ps.ori2cur[p_j0]
                xji0  = self.ps.x0[p_j] - self.ps.x0[p_i]
                r = xji0.norm()
                self.K[p_i0][j] = self.ps.m_V0[p_j] * self.W(r, self.ps.support_radius) / (r*r + eps)


    @ti.kernel 
    def compute_F(self, x: ti.template()):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            Ds_i = ti.math.mat3(0.0)
            # Dm = ti.math.mat3(0.0)
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xji0 = self.ps.x0[p_j] - self.ps.x0[p_i]
                xji = x[p_j] - x[p_i]
                # Ds_i += xji.outer_product(xji0)
                Ds_i -= self.ps.m_V0[p_j] * xji.outer_product(self.gradW(xji0, self.ps.support_radius))

            self.F[p_i] = Ds_i @ self.L[p_i0]
            # print(f"F: {self.F[p_i]}")


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
        mu = YM / (2.0 * (1.0 + PR))
        lamb = 2.0 * mu * PR / (1.0 - 2.0 * PR)

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            F_i = self.F[p_i]
            J_i = ti.math.determinant(F_i)
            # F_i = ti.Matrix.identity(float, 3)
            U, sig, V = self.ssvd(F_i)
            R_i = U @ V.transpose()

            self.P_s[p_i] = 2.0 * mu * (F_i - R_i)
            self.P_v[p_i] = ti.Matrix.identity(float, 3)
            if J_i > 1.0:
                self.P_v[p_i] = lamb * (J_i - 1.0) * ti.Matrix.identity(float, 3) @ R_i


    @ti.kernel
    def compute_ZE(self, alpha: float, YM: float, PR: float, x:ti.template()):
        for p in ti.grouped(self.ZE):
            self.ZE[p] = ti.math.vec3(0.0)

        mu = YM / (2.0 * (1.0 + PR))

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            F_i = self.F[p_i]
            K_i = self.K[p_i0]
            LgradW_i = self.LgradW[p_i0]

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]

                xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
                xij = x[p_i] - x[p_j]
                e_ij = F_i @ xij0 - xij
                
                sumVs = 0.0
                for k in range(self.ps.solid_neighbors_num[p_i0]):
                    p_k0 = self.ps.solid_neighbors[p_i0, k]
                    p_k = self.ps.ori2cur[p_k0]
                    s = (LgradW_i[0, k] * xij0[0] +
                        LgradW_i[1, k] * xij0[1] +
                        LgradW_i[2, k] * xij0[2])
                    sumVs += self.ps.m_V0[p_k] * s
                    ti.atomic_add(self.ZE[p_k], self.ps.m_V0[p_k] * s * e_ij * K_i[j])

                ti.atomic_add(self.ZE[p_j],  e_ij * K_i[j])
                ti.atomic_add(self.ZE[p_i], -(1.0 + sumVs) * e_ij * K_i[j])

        for p in ti.grouped(self.ZE):
            self.ZE[p] *= alpha * mu
    
    @ti.kernel
    def compute_gradient(self, dt: float):
        # dtSq = dt ** 2
        for p_i in ti.grouped(self.ps.x):
            self.grad[p_i] = ti.math.vec3(0.0)
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            PLs_i = self.P_s[p_i] @ self.L[p_i0]
            PLv_i = self.P_v[p_i] @ self.L[p_i0]

            f_s = ti.math.vec3(0.0)
            f_ve = ti.math.vec3(0.0)

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]

                xji0 = self.ps.x0[p_j] -self.ps.x0[p_i]
                PLs_j = self.P_s[p_j] @ self.L[p_j0]
                PLv_j = self.P_v[p_j] @ self.L[p_j0]

                f_s += self.ps.m_V0[p_j] * (PLs_i + PLs_j) @ self.gradW(xji0, self.ps.support_radius)
                # f_s += (PL_i + PL_j) @ xji0
                f_ve += self.ps.m_V0[p_j] * (PLv_i + PLv_j) @ self.gradW(xji0, self.ps.support_radius)

            self.grad[p_i] = self.ps.m_V0[p_i] * dt * (f_s + f_ve + self.ZE[p_i])



    def PCG(self, x, b):

        pass


    @ti.kernel
    def compute_Ax(self, Ax: ti.template(), x: ti.template()):

        pass

    def solve(self, alpha: float, YM: float, PR: float, dt: float):

        add(self.x_tmp, self.ps.x, dt, self.ps.v)
        self.compute_F(self.x_tmp)

        self.compute_ZE(alpha, YM, PR, self.x_tmp)
        self.compute_P(YM, PR)
        self.compute_gradient(dt)

        # TODO: v +=  M^-1 * (grad)
        self.v_tmp.copy_from(self.ps.v)
        coef_wise_mul(self.a, self.ps.m_inv, self.grad)
        add(self.v_tmp, self.v_tmp, -1.0, self.a)
        self.ps.v.copy_from(self.v_tmp)
        # self.PCG(x=None, b=None)