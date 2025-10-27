import time

from taichi.lang.matrix_ops import diag
from math_utils   import *
from sph_kernel   import *
from elastic_util import *

@ti.data_oriented
class Elasticity:
    def __init__(self, particle_system):

        self.ps = particle_system

        #TODO: allocate F, L for deformable particles only
        self.L      = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.F      = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.dJdF   = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.F_tmp  = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.P      = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.Q      = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.ps.particle_max_num, 9)) # eigenvectors of volume hessian per paticle
        self.D      = ti.Matrix.field(n=3, m=3, dtype=float, shape=6)

        self.J      = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.lamb   = ti.field(dtype=float, shape=(self.ps.particle_max_num, 9)) # eigenvalues of volume hessian per paticle
        
        self.K      = ti.Vector.field(n=self.ps.cache_size, dtype=float, shape=self.ps.particle_max_num)
        self.LgradW = ti.Vector.field(3, dtype=float, shape=(self.ps.particle_max_num, self.ps.cache_size))
        self.Aii    = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)

        self.ZE     = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.grad   = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.x_tmp  = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.v_tmp  = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.a      = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        # self.d2EdF2 = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.ps.particle_max_num, 4, 4))
        self.gradW  = spiky_kernel_derivative
        self.W      = cubic_kernel

        self.Ap    = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.b_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.z_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.r_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.p_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)

        self.max_iteration_pcg = 10
        self.tol_pcg           = 4

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
                if self.ps.object_id[p_j] != self.ps.object_id[p_i]:
                    continue
                p_j0 = self.ps.cur2ori[p_j]
                if self.ps.solid_neighbors_num[p_i0] < self.ps.cache_size:
                    self.ps.solid_neighbors[p_i0, self.ps.solid_neighbors_num[p_i0]] = p_j0
                    self.ps.solid_neighbors_num[p_i0] += 1

            # print(self.ps.solid_neighbors_num[p_i0])

        # compute rest volume
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

        for k in ti.grouped(self.ps.x_s):
            tmp = 0.0
            X_k = self.ps.x_0_s[k]
            for j in range(self.ps.surface_neighbor_num[k]):
                p_j = self.ps.surface_neighbor_idx[k, j]
                # if self.ps.object_id[p_j] != self.ps.surface_vertex_object_id[k]:
                #     continue
                p_j0 = self.ps.cur2ori[p_j]
                X_kj = X_k - self.ps.x0[p_j]
                tmp += self.ps.m_V0[p_j] * self.W(X_kj.norm(), self.ps.support_radius)
                self.ps.surface_neighbor_idx[k, j] = p_j0

            self.ps.skinning_weight[k] = 1.0 / tmp
                
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
                p_k = self.ps.ori2cur[p_k0]
                xik0 = self.ps.x0[p_i] - self.ps.x0[p_k]
                self.LgradW[p_i0, k] = Li @ self.gradW(xik0, self.ps.support_radius)

            # computeKze
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j  = self.ps.ori2cur[p_j0]
                xji0  = self.ps.x0[p_j] - self.ps.x0[p_i]
                r = xji0.norm()
                self.K[p_i0][j] = self.ps.m_V0[p_j] * self.W(r, self.ps.support_radius) / (r*r + eps)


        # D6 ~ D8 in Appendix A
        self.D[0] = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
        self.D[1] = ti.math.mat3([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        self.D[2] = ti.math.mat3([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.D[3] = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.D[4] = ti.math.mat3([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        self.D[5] = ti.math.mat3([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

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

        #compute deformation gradient F
        # for p_i in ti.grouped(self.ps.x):
        #     if self.ps.material[p_i] != self.ps.material_solid:
        #         continue
        #
        #     p_i0 = self.ps.cur2ori[p_i]
        #     Ds_i = ti.math.mat3(0.0)
        #     # Dm = ti.math.mat3(0.0)
        #     for j in range(self.ps.solid_neighbors_num[p_i0]):
        #         p_j0 = self.ps.solid_neighbors[p_i0, j]
        #         p_j = self.ps.ori2cur[p_j0]
        #         xji0 = self.ps.x0[p_j] - self.ps.x0[p_i]
        #         xji = x[p_j] - x[p_i]
        #         Ds_i -= self.ps.m_V0[p_j] * xji.outer_product(self.gradW(xji0, self.ps.support_radius))
        #
        #     self.F[p_i] = Ds_i @ self.L[p_i0]

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            F_i = self.F[p_i]
            self.J[p_i] = ti.math.determinant(F_i)

            # F_i = ti.Matrix.identity(float, 3)
            U, sig, V = self.ssvd(F_i)

            s0 = sig[0, 0]
            s1 = sig[1, 1]
            s2 = sig[2, 2]

            R_i = U @ V.transpose()
            self.dJdF[p_i] = compute_dJdF_3x3(F_i)
            # self.P[p_i] = 2.0 * mu * (F_i - R_i) + lamb * (self.J[p_i] - 1.0) * self.dJdF[p_i]
            self.P[p_i] = 2.0 * mu * (F_i - R_i)


            # volume hessian computation according to Stable Neo Hookean [Smith et al. 2018]

            # Eq.(32)
            self.lamb[p_i, 3] = s0
            self.lamb[p_i, 4] = s1
            self.lamb[p_i, 5] = s2

            self.lamb[p_i, 6] = -s0
            self.lamb[p_i, 7] = -s1
            self.lamb[p_i, 8] = -s2
            sigv = ti.Vector([s0, s1, s2])
            #Appendix A, Eq.(62)
            for k in range(6):
                self.Q[p_i, k + 3] = ti.rsqrt(2) *  U @ self.D[k] @ V.transpose()

            # Eq(39)~Eq(40)
            I_C = s0 * s0 + s1 * s1 + s2 * s2
            I_C = ti.max(I_C, 1e-12)

            t = 2.0 * ti.math.sqrt(I_C/3.0)
            u = (3.0 * self.J[p_i]/I_C) * ti.math.sqrt(3.0/I_C)
            arg = ti.min(1.0, ti.max(-1.0, u))

            for k in range(3):
                phi = ti.acos(arg) + 2.0 * ti.math.pi * k
                lamb_k = t * ti.cos(phi / 3.0)
                self.lamb[p_i, k] = lamb_k
                D_k = ti.math.mat3([[s0 * s2 + s1 * self.lamb[p_i, k], 0, 0], [0, s1 * s2 + s0 * self.lamb[p_i, k], 0], [0, 0, self.lamb[p_i, k] ** 2 - s2 ** 2]])
                # a = sigv[k]
                # b = sigv[(k + 1) % 3]
                # c = sigv[(k + 2) % 3]

                # D_k = ti.math.mat3([
                #     [a * c + b * lamb_k, 0.0, 0.0],
                #     [0.0, b * c + a * lamb_k, 0.0],
                #     [0.0, 0.0, lamb_k * lamb_k - c * c],
                # ])
                q_k = ti.math.sqrt(double_dot_product(D_k, D_k))
                self.Q[p_i, k] = (1.0 / q_k) * U @ D_k @ V.transpose()


            # self.P_v[p_i] = ti.Matrix.identity(float, 3)
            # if J_i > 1.0:
            #     self.P[p_i] += lamb * (J_i - 1.0) * ti.Matrix.identity(float, 3) @ R_i


    @ti.kernel
    def compute_ZE(self, alpha: float, YM: float, PR: float, x:ti.template()):
        for p in ti.grouped(self.ZE):
            self.ZE[p] = ti.Vector.zero(float, 3)
        
        mu = YM / (2.0 * (1.0 + PR))
        lamb = 2.0 * mu * PR / (1.0 - 2.0 * PR)

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue
            p_i0 = self.ps.cur2ori[p_i]
            F_i = self.F[p_i]
            K_i = self.K[p_i0]

            grad_sum = ti.math.vec3(0.0)
            for k in range(self.ps.solid_neighbors_num[p_i0]):
                p_k0 = self.ps.solid_neighbors[p_i0, k]
                p_k = self.ps.ori2cur[p_k0]
                grad_sum += self.ps.m_V0[p_k] * self.LgradW[p_i0, k]

            accum = ti.math.mat3(0.0)
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
                xij = x[p_i] - x[p_j]
                e_ij = F_i @ xij0 - xij
                E_ij = e_ij * K_i[j]
                sumVs = grad_sum.dot(xij0)
                ti.atomic_add(self.ZE[p_j], E_ij)
                ti.atomic_add(self.ZE[p_i], -(1.0 + sumVs) * E_ij)
                accum += E_ij.outer_product(xij0)
                    
            for k in range(self.ps.solid_neighbors_num[p_i0]):
                p_k0 = self.ps.solid_neighbors[p_i0, k]
                p_k = self.ps.ori2cur[p_k0]
                ti.atomic_add(self.ZE[p_k], self.ps.m_V0[p_k] * (accum @ self.LgradW[p_i0, k]))
                
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
            f_i = ti.math.vec3(0.0)

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]

                xji0 = self.ps.x0[p_j] -self.ps.x0[p_i]
                PL_i = - self.P[p_i] @ self.LgradW[p_i0, j]
                PL_j = self.P[p_j] @ self.L[p_j0] @ self.gradW(xji0, self.ps.support_radius)

                f_i += self.ps.m_V0[p_j] * (PL_i + PL_j)


            self.grad[p_i] = self.ps.m_V0[p_i] * dt * (f_i + self.ZE[p_i])

    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> float:

        ret = 0.0
        for p_i in ti.grouped(self.ps.x):

            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            ret += ti.math.dot(a[p_i], b[p_i])

        return ret

    @ti.kernel
    def add(self, ret: ti.template(), v0: ti.template(), scale: float, v1: ti.template()):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            ret[p_i] = v0[p_i] + scale * v1[p_i]

    @ti.kernel
    def compute_Aii(self, alpha: float,YM: float, PR: float, dt: float):
        mu = YM / (2.0 * (1.0 + PR))
        eps = 1e-12
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            Aii = ti.Matrix.identity(float, 3) * self.ps.m[p_i]   # M_ii

            # Stretch
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j  = self.ps.ori2cur[p_j0]
                g    = self.LgradW[p_i0, j]
                Aii   += 2.0 * dt * dt * mu * self.ps.m_V0[p_i] * self.ps.m_V0[p_j] * g.outer_product(g)

            # K_i = 0.0
            # for j in range(self.ps.solid_neighbors_num[p_i0]):
            #     K_i += self.K[p_i0][j]
            # A += dt * dt * (alpha * mu) * K_i * ti.Matrix.identity(float, 3)

            self.Aii[p_i] = Aii + eps

    @ti.kernel
    def apply_preconditioner(self, z: ti.template(), r: ti.template()):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue
            
            # z[p_i] = self.Aii[p_i] @ r[p_i]
            z[p_i] = r[p_i]


    def PCG(self, x, b, alpha, YM, PR, dt):

            x.fill(0.0)
            r = self.r_pcg
            z = self.z_pcg
            p = self.p_pcg
            Ap = self.Ap
            # Jp = self.dp

            self.compute_Ax(Ap, x, alpha, YM, PR, dt)
            # self.add(r, b, -1.0, Ap)
            r.copy_from(b)
            self.apply_preconditioner(z, r)

            rz_old = self.dot(r, z)
            # r_old = self.dot(r, r)
            pcg_iter = 0
            # if r_old > pow(10, -self.tol_pcg):
            if rz_old > 1e-12:
                p.copy_from(z)
                iter = 0
                for _ in range(self.max_iteration_pcg):
                    # Ap.fill(0.0)
                    self.compute_Ax(Ap, p, alpha, YM, PR, dt)
                    pAp = self.dot(p, Ap)
                    if pAp < 0.0:
                        print("Warning: non-positive definite matrix!")

                    alpha = rz_old / pAp

                    self.add(x, x, alpha, p)
                    self.add(r, r, -alpha, Ap)
                    self.apply_preconditioner(z, r)

                    pcg_iter += 1
                    r_new = self.dot(r, r)
                    # if self.print_pcg_error:
                    #     print(f"PCG error: {r_new}")

                    if r_new < pow(10, -self.tol_pcg) or pcg_iter >= self.max_iteration_pcg:
                        print("PCG ITER:", iter)
                        break

                    iter += 1
                    rz_new = self.dot(r, z)
                    beta = rz_new / rz_old
                    self.add(p, z, beta, p)
                    rz_old = rz_new

            # Collect PCG iteration count per PCG solve
            # self.pcg_total_iter += pcg_iter

    @ti.kernel
    def compute_Ax(self, Ax: ti.template(), x: ti.template(), alpha: float, YM: float, PR: float, dt:float):

        for p in ti.grouped(self.ZE):
            self.ZE[p] = ti.math.vec3(0.0)

        mu = YM / (2.0 * (1.0 + PR))
        lamb = 2.0 * mu * PR / (1.0 - 2.0 * PR)

        #step 1 KDx
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            Ds_i = ti.math.mat3(0.0)

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xji0 = self.ps.x0[p_j] - self.ps.x0[p_i]
                xji = x[p_j] - x[p_i]
                Ds_i -= self.ps.m_V0[p_j] * xji.outer_product(self.gradW(xji0, self.ps.support_radius))

            self.F_tmp[p_i] = Ds_i @ self.L[p_i0]

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            dJdF_i = self.dJdF[p_i]
            a = double_dot_product(dJdF_i, self.F_tmp[p_i])

            test = ti.math.mat3(0.0)
            # for k in range(9):
            #     lambJ = self.lamb[p_i, k] * (self.J[p_i] - 1.0)
            #     lambJ = ti.max(lambJ, 0.0)
            #     test += lambJ * double_dot_product(self.Q[p_i, k], self.F_tmp[p_i]) * self.Q[p_i, k]

            # self.P[p_i] = 2.0 * mu * self.F_tmp[p_i] + lamb * (a * dJdF_i + test)
            self.P[p_i] = 2.0 * mu * self.F_tmp[p_i]

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            K_i = self.K[p_i0]

            grad_sum = ti.math.vec3(0.0)
            for k in range(self.ps.solid_neighbors_num[p_i0]):
                p_k0 = self.ps.solid_neighbors[p_i0, k]
                p_k = self.ps.ori2cur[p_k0]
                grad_sum += self.ps.m_V0[p_k] * self.LgradW[p_i0, k]

            accum = ti.math.mat3(0.0)
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
                xij = x[p_i] - x[p_j]
                e_ij = self.F_tmp[p_i] @ xij0 - xij
                E_ij = e_ij * K_i[j]
                sumVs = grad_sum.dot(xij0)
                ti.atomic_add(self.ZE[p_j], E_ij)
                ti.atomic_add(self.ZE[p_i], -(1.0 + sumVs) * E_ij)
                accum += E_ij.outer_product(xij0)
                    
            for k in range(self.ps.solid_neighbors_num[p_i0]):
                p_k0 = self.ps.solid_neighbors[p_i0, k]
                p_k = self.ps.ori2cur[p_k0]
                ti.atomic_add(self.ZE[p_k], self.ps.m_V0[p_k] * (accum @ self.LgradW[p_i0, k]))
                
        for p in ti.grouped(self.ZE):
            self.ZE[p] *= alpha * mu

        # step 2 Mx + dt ** 2 * D^TKDx
        for p_i in ti.grouped(self.ps.x):

            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            f_i = ti.math.vec3(0.0)

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]

                xji0 = self.ps.x0[p_j] - self.ps.x0[p_i]
                PL_i = - self.P[p_i] @ self.LgradW[p_i0, j]
                PL_j = self.P[p_j] @ self.L[p_j0] @ self.gradW(xji0, self.ps.support_radius)

                f_i += self.ps.m_V0[p_j] * (PL_i + PL_j)
            Ax[p_i] = self.ps.m[p_i] * x[p_i] + self.ps.m_V0[p_i] * dt ** 2 * (f_i + self.ZE[p_i])


    @ti.kernel
    def compute_xAx(self, x: ti.template(), alpha: float, YM: float, PR: float, dt: float) -> float:


        return 0.0


    def solve(self, alpha, YM, PR, dt):

        add(self.x_tmp, self.ps.x, dt, self.ps.v)

        self.compute_F(self.x_tmp)
        self.compute_P(YM, PR)
        self.compute_ZE(alpha, YM, PR, self.x_tmp)
        self.compute_gradient(dt)
        self.compute_Aii(alpha, YM, PR, dt)

        self.v_tmp.copy_from(self.ps.v)
        self.PCG(x=self.a, b=self.grad, alpha=alpha, YM=YM, PR=PR, dt=dt)
        add(self.v_tmp, self.v_tmp, -1.0, self.a)
        self.ps.v.copy_from(self.v_tmp)

    @ti.kernel
    def update_surface_vertex(self):

        for k in ti.grouped(self.ps.x_s):
            x_k = ti.math.vec3(0.0)
            X_k = self.ps.x_0_s[k]
            s_k = self.ps.skinning_weight[k]

            for j in range(self.ps.surface_neighbor_num[k]):
                j0   = self.ps.surface_neighbor_idx[k, j]
                p_j  = self.ps.ori2cur[j0]
                X_kj = X_k - self.ps.x0[p_j]
                x_k += s_k * self.ps.m_V0[p_j] * (self.F[p_j] @ X_kj + self.ps.x[p_j]) * self.W(X_kj.norm(), self.ps.support_radius)

            self.ps.x_s[k] = x_k

    def apply_mesh_skinning(self):

        self.compute_F(self.ps.x)
        self.update_surface_vertex()

