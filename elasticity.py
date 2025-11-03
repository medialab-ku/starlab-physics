import time

from numpy import dtype
from taichi import atomic_add
from taichi.lang.matrix_ops import diag
from math_utils   import *
from sph_kernel   import *
from elastic_util import *

@ti.data_oriented
class Elasticity:
    def __init__(self, particle_system):

        self.ps = particle_system
        self.k = 1e6

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
        
        self.K      = ti.field(dtype=float, shape=(self.ps.particle_max_num, self.ps.cache_size))


        self.test1   = ti.field(dtype=float, shape=(self.ps.particle_max_num, self.ps.cache_size))
        self.test2   = ti.field(dtype=float, shape=(self.ps.particle_max_num, self.ps.cache_size))
        self.VjLigradW = ti.Vector.field(3, dtype=float, shape=(self.ps.particle_max_num, self.ps.cache_size))
        self.VjLjgradW = ti.Vector.field(3, dtype=float, shape=(self.ps.particle_max_num, self.ps.cache_size))
        self.invAii     = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.Aii     = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
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

        self.max_iteration_pcg = 3
        self.tol_pcg           = 5
        self.precondition      = 0

        self.pcg_last_iter = 0
        self.stats_elapsed_ms = []
        self.stats_pcg_iter = []

        self.eij_num = ti.field(dtype=int, shape=1)
        self.eij_pair = ti.Vector.field(n=2, dtype=int, shape=(self.ps.particle_max_num * self.ps.cache_size))
        self.eij_sk = ti.field(dtype=float, shape=(self.ps.particle_max_num * self.ps.cache_size, self.ps.cache_size))
        self.K = ti.field(dtype=float, shape=(self.ps.particle_max_num * self.ps.cache_size))



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
                if self.ps.object_id[p_j] != self.ps.surface_vertex_object_id[k]:
                    continue
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
                xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
                # Dm += xji0.outer_product(xji0)
                Dm += self.ps.m_V0[p_j] * self.gradW(xij0, self.ps.support_radius).outer_product(-xij0)
            
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
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                Lj = self.L[p_j0]
                xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
                self.VjLigradW[p_i0, j] = self.ps.m_V0[p_j] * Li @ self.gradW(xij0, self.ps.support_radius)
                self.VjLjgradW[p_i0, j] = self.ps.m_V0[p_j] * Lj @ self.gradW(xij0, self.ps.support_radius)

        num = 0
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
                n = atomic_add(num, 1)
                self.eij_pair[n] = ti.math.ivec2([p_i0, j])
                self.K[n] = self.ps.m_V0[p_i] * self.ps.m_V0[p_j] * self.W(xij0.norm(), self.ps.support_radius) / xij0.norm_sqr()

                sum = 0.0
                for k in range(self.ps.solid_neighbors_num[p_i0]):
                    self.eij_sk[n, k] = self.VjLigradW[p_i0, k].dot(xij0)
                    p_k0 = self.ps.solid_neighbors[p_i0, k]
                    if p_j0 == p_k0:
                        self.eij_sk[n, k] += 1.0
                    sum += self.eij_sk[n, k]

                self.eij_sk[n, self.ps.solid_neighbors_num[p_i0] + 1] = sum + 1.0

        self.eij_num[0] = num

        for p_i in ti.grouped(self.ps.x):

            p_i0 = self.ps.cur2ori[p_i]
            grad_sum = ti.math.vec3(0.0)
            # dF_i / dx_i
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                grad_sum += self.VjLigradW[p_i0, j]

            # (sum_j e_ij * K_ij * Xij0_T)  V_j * Li * gradW
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
                ViLjgradW = self.ps.m_V0[p_i] * self.VjLjgradW[p_i0, j] / self.ps.m_V0[p_j]
                self.test1[p_i0, j] = grad_sum.dot(xij0)  # dF_i/dx_i : Xij0
                self.test2[p_i0, j] = - ViLjgradW.dot(xij0)

        # D6 ~ D8 in Appendix A
        self.D[0] = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],  [0.0, -1.0, 0.0]])
        self.D[1] = ti.math.mat3([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0],  [-1.0, 0.0, 0.0]])
        self.D[2] = ti.math.mat3([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        self.D[3] = ti.math.mat3([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0],  [0.0, 1.0, 0.0]])
        self.D[4] = ti.math.mat3([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0],  [1.0, 0.0, 0.0]])
        self.D[5] = ti.math.mat3([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0],  [0.0, 0.0, 0.0]])

    @ti.kernel
    def compute_F(self, x: ti.template()):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            self.F[p_i] = ti.math.mat3(0.0)
            p_i0 = self.ps.cur2ori[p_i]
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xji = x[p_j] - x[p_i]
                self.F[p_i] += xji.outer_product(self.VjLigradW[p_i0, j])


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
            # self.J[p_i] = ti.math.determinant(F_i)

            # F_i = ti.Matrix.identity(float, 3)
            U, sig, V = self.ssvd(F_i)

            # s0 = sig[0, 0]
            # s1 = sig[1, 1]
            # s2 = sig[2, 2]

            R_i = U @ V.transpose()
            self.P[p_i] = 2.0 * mu * (F_i - R_i)

    @ti.kernel
    def compute_ZE(self, alpha: float, YM: float, PR: float, x:ti.template()):

        mu = YM / (2.0 * (1.0 + PR))
        lamb = 2.0 * mu * PR / (1.0 - 2.0 * PR)

        for t in range(self.eij_num[0]):

            ij = self.eij_pair[t]
            p_i0, j = ij[0], ij[1]

            p_i = self.ps.ori2cur[p_i0]
            p_j0 = self.ps.solid_neighbors[p_i0, j]
            p_j = self.ps.ori2cur[p_j0]

            xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
            xij = x[p_i] - x[p_j]
            F_i = self.F[p_i]
            Kij_e_ij = alpha * mu * self.K[t] * (F_i @ xij0 - xij)

            for k in range(self.ps.solid_neighbors_num[p_i0]):
                p_k0 = self.ps.solid_neighbors[p_i0, k]
                p_k = self.ps.ori2cur[p_k0]
                sk = self.eij_sk[t, k]
                self.ZE[p_k] += sk * Kij_e_ij

            self.ZE[p_i] -= self.eij_sk[t, self.ps.solid_neighbors_num[p_i0] + 1] * Kij_e_ij


            # self.ZE[p_i] *= alpha * mu
    @ti.kernel
    def compute_gradient(self, x: ti.template(), dt: float):
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

                PL_i = -self.P[p_i] @ self.VjLigradW[p_i0, j]
                PL_j = -self.P[p_j] @ self.VjLjgradW[p_i0, j]
                f_i += (PL_i + PL_j)

            self.grad[p_i] = self.ps.m_V0[p_i] * dt * f_i + dt * self.ZE[p_i]
            if self.ps.is_pinned[p_i]:
                self.grad[p_i] += dt * self.k * (x[p_i] - self.ps.x0[p_i])


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
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            mI = ti.Matrix.identity(float, 3) * self.ps.m[p_i]   # M_ii
            
            Aii = mI
            hh = 0.0
            g_sum = ti.math.vec3(0.0)
            # Stretch
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j  = self.ps.ori2cur[p_j0]
                g_sum += self.VjLigradW[p_i0, j]
                h = self.VjLjgradW[p_i0, j]
                hh += h.dot(h) / self.ps.m_V0[p_j]

            Aii += 2.0 * dt * dt * mu * self.ps.m_V0[p_i] * (g_sum.dot(g_sum) + self.ps.m_V0[p_i] * hh)
            self.Aii[p_i] = Aii

        for t in range(self.eij_num[0]):
            # Aii += dt * dt * alpha * mu * self.K[t] * ti.Matrix.identity(float, 3)
            ij = self.eij_pair[t]
            p_i0, j = ij[0], ij[1]
            p_i = self.ps.ori2cur[p_i0]
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            s_idx = self.ps.solid_neighbors_num[p_i0] + 1
            sk = self.eij_sk[t, s_idx]
            self.Aii[p_i] += dt * dt * alpha * mu * self.K[t] * (sk - 1.0) * (sk - 1.0) * ti.Matrix.identity(float, 3)
            self.invAii[p_i] = (self.Aii[p_i]).inverse()


    @ti.kernel
    def apply_preconditioner(self, z: ti.template(), r: ti.template()):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            if self.precondition == 1:
                z[p_i] = ti.Matrix.identity(float, 3) * self.ps.m_inv[p_i] @ r[p_i]
            elif self.precondition == 2:
                z[p_i] = self.invAii[p_i] @ r[p_i]
            else:
                z[p_i] = r[p_i]


    def PCG(self, x, b, alpha, YM, PR, dt):

            x.fill(0.0)
            r = self.r_pcg
            z = self.z_pcg
            p = self.p_pcg
            Ap = self.Ap

            r.copy_from(b)
            self.apply_preconditioner(z, r)
            rz_old = self.dot(r, z)
            pcg_iter = 0
            if rz_old > 1e-12:
                p.copy_from(z)
                iter = 0
                for _ in range(self.max_iteration_pcg):

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

                self.pcg_last_iter = pcg_iter

            # Collect PCG iteration count per PCG solve
            # self.pcg_total_iter += pcg_iter


    @ti.kernel
    def compute_Ax(self, Ax: ti.template(), x: ti.template(), alpha: float, YM: float, PR: float, dt:float):


        mu = YM / (2.0 * (1.0 + PR))
        lamb = 2.0 * mu * PR / (1.0 - 2.0 * PR)

        #step 1 KDx
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            F_i = ti.math.mat3(0.0)

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xji = x[p_j] - x[p_i]
                F_i += xji.outer_product(self.VjLigradW[p_i0, j])

            self.F_tmp[p_i] = F_i


        for p_i in ti.grouped(self.ps.x):
            Ax[p_i] = self.ps.m[p_i] * x[p_i]

        for t in range(self.eij_num[0]):

            ij = self.eij_pair[t]
            p_i0, j = ij[0], ij[1]

            p_i = self.ps.ori2cur[p_i0]
            p_j0 = self.ps.solid_neighbors[p_i0, j]
            p_j = self.ps.ori2cur[p_j0]

            xij0 = self.ps.x0[p_i] - self.ps.x0[p_j]
            xij = x[p_i] - x[p_j]
            F_i = self.F[p_i]
            Kij_e_ij = dt ** 2 * alpha * mu * self.K[t] * (F_i @ xij0 - xij)

            for k in range(self.ps.solid_neighbors_num[p_i0]):
                p_k0 = self.ps.solid_neighbors[p_i0, k]
                p_k = self.ps.ori2cur[p_k0]
                sk = self.eij_sk[t, k]
                Ax[p_k] += sk * Kij_e_ij

            Ax[p_i] -= self.eij_sk[t, self.ps.solid_neighbors_num[p_i0] + 1] * Kij_e_ij

        # step 2 Mx + dt ** 2 * D^TKDx
        for p_i in ti.grouped(self.ps.x):

            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            f_i = ti.math.vec3(0.0)

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                # 2.0 * mu * self.F_tmp[p_i]
                PL_i = -self.F_tmp[p_i] @ self.VjLigradW[p_i0, j]
                PL_j = -self.F_tmp[p_j] @ self.VjLjgradW[p_i0, j]
                f_i += 2.0 * mu * (PL_i + PL_j)

            Ax[p_i] += self.ps.m_V0[p_i] * dt ** 2 * f_i

            if self.ps.is_pinned[p_i]:
                Ax[p_i] += dt * dt * self.k * x[p_i]


    def solve(self, alpha, YM, PR, dt):

        # t0 = time.perf_counter()
        add(self.x_tmp, self.ps.x, dt, self.ps.v)
        self.compute_F(self.x_tmp)
        # elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # print("deformation gradient: ", elapsed_ms)

        self.compute_P(YM, PR)

        # t0 = time.perf_counter()

        self.ZE.fill(0.0)
        self.compute_ZE(alpha, YM, PR, self.x_tmp)
        # elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # print("zero energy: ", elapsed_ms)
        self.compute_gradient(self.x_tmp, dt)
        self.compute_Aii(alpha, YM, PR, dt)

        # t0 = time.perf_counter()
        self.v_tmp.copy_from(self.ps.v)
        self.PCG(x=self.a, b=self.grad, alpha=alpha, YM=YM, PR=PR, dt=dt)
        add(self.v_tmp, self.v_tmp, -1.0, self.a)
        self.ps.v.copy_from(self.v_tmp)
        # elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # print("linear solve: ", elapsed_ms)

        # self.stats_elapsed_ms.append(float(elapsed_ms))
        self.stats_pcg_iter.append(int(self.pcg_last_iter))


    

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

    def clear_stats(self):
        # self.stats_elapsed_ms.clear()
        self.stats_pcg_iter.clear()