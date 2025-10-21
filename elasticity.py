import time
from math_utils import *
from sph_kernel import *

@ti.data_oriented
class Elasticity:
    def __init__(self, particle_system):

        self.ps = particle_system

        #TODO: allocate F, L for deformable particles only
        self.L     = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.F     = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.F_tmp = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.P = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.K = ti.Vector.field(n=self.ps.cache_size, dtype=float, shape=self.ps.particle_max_num)
        self.LgradW = ti.Vector.field(3, dtype=float, shape=(self.ps.particle_max_num, self.ps.cache_size))

        self.ZE = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.grad  = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.x_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.a = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.d2EdF2 = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.ps.particle_max_num, 4, 4))
        self.gradW = spiky_kernel_derivative
        self.W = cubic_kernel

        self.Ap    = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.b_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.z_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.r_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.p_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)

        self.max_iteration_pcg = 1000
        self.tol_pcg = 3

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

            # computeK
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j  = self.ps.ori2cur[p_j0]
                xji0  = self.ps.x0[p_j] - self.ps.x0[p_i]
                r = xji0.norm()
                self.K[p_i0][j] = self.ps.m_V0[p_j] * self.W(r, self.ps.support_radius) / (r*r + eps)



        for k in ti.grouped(self.ps.x_s):
        
            tmp = 0.0
            X_k = self.ps.x_0_s[k]
            for j in range(self.ps.surface_neighbor_num[k]):
                j0 = self.ps.surface_neighbor_idx[k, j]
                p_j = self.ps.ori2cur[j0]
                X_kj = X_k - self.ps.x0[p_j]
                tmp += self.ps.m_V0[p_j] * self.W(X_kj.norm(),self.ps.support_radius)
        
            self.ps.skinning_weight[k] = 1.0 / tmp

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
    def compute_P(self, x: ti.template(), YM: float, PR: float):

        # stretch term, volume term
        mu = YM / (2.0 * (1.0 + PR))
        lamb = 2.0 * mu * PR / (1.0 - 2.0 * PR)

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

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            F_i = self.F[p_i]
            J_i = ti.math.determinant(F_i)
            # F_i = ti.Matrix.identity(float, 3)
            U, sig, V = self.ssvd(F_i)
            R_i = U @ V.transpose()

            self.P[p_i] = 2.0 * mu * (F_i - R_i)
            # self.P_v[p_i] = ti.Matrix.identity(float, 3)
            # if J_i > 1.0:
            #     self.P[p_i] += lamb * (J_i - 1.0) * ti.Matrix.identity(float, 3) @ R_i


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
                    g = self.LgradW[p_i0, k]
                    s = ti.math.dot(g, xij0)
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
            # PL_i = self.P[p_i] @ self.L[p_i0]
            # PLv_i = self.P_v[p_i] @ self.L[p_i0]

            f_i = ti.math.vec3(0.0)
            # f_ve = ti.math.vec3(0.0)

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]

                xji0 = self.ps.x0[p_j] -self.ps.x0[p_i]
                PL_i = - self.P[p_i] @ self.LgradW[p_i0, j]
                PL_j = self.P[p_j] @ self.L[p_j0] @ self.gradW(xji0, self.ps.support_radius)
                # PL_j = self.P[p_j] @ self.L[p_j0]
                # PLv_j = self.P_v[p_j] @ self.L[p_j0]

                f_i += self.ps.m_V0[p_j] * (PL_i + PL_j)


            # self.grad[p_i] = self.ps.m_V0[p_i] * dt * (f_s + f_ve + self.ZE[p_i])
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


    def CG(self, x, b, alpha, YM, PR, dt):

            x.fill(0.0)
            r = self.r_pcg
            p = self.p_pcg
            Ap = self.Ap
            # Jp = self.dp

            self.compute_Ax(Ap, x, alpha, YM, PR, dt)
            self.add(r, b, -1.0, Ap)

            r_old = self.dot(r, r)
            pcg_iter = 0
            # if r_old > pow(10, -self.tol_pcg):
            if r_old > 1e-12:
                p.copy_from(r)
                iter = 0
                for _ in range(self.max_iteration_pcg):
                    # Ap.fill(0.0)
                    self.compute_Ax(Ap, p, alpha, YM, PR, dt)

                    pAp = self.dot(p, Ap)
                    if pAp < 0.0:
                        print("Warning: non-positive definite matrix!")

                    alpha = r_old / pAp

                    self.add(x, x, alpha, p)
                    self.add(r, r, -alpha, Ap)

                    pcg_iter += 1
                    r_new = self.dot(r, r)
                    # if self.print_pcg_error:
                    #     print(f"PCG error: {r_new}")

                    if r_new < pow(10, -self.tol_pcg) or pcg_iter >= self.max_iteration_pcg:
                        print("PCG ITER", iter)
                        break

                    iter += 1
                    beta = r_new / r_old
                    self.add(p, r, beta, p)
                    r_old = r_new

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
            Ds_i = ti.math.mat3(0.0)
            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]
                xji0 = self.ps.x0[p_j] - self.ps.x0[p_i]
                xji = x[p_j] - x[p_i]
                Ds_i -= self.ps.m_V0[p_j] * xji.outer_product(self.gradW(xji0, self.ps.support_radius))

            self.P[p_i] = 2.0 * mu * Ds_i @ self.L[p_i0]

        #TODO: H^TKHx
        for p in ti.grouped(self.ZE):
            self.ZE[p] = ti.math.vec3(0.0)

        mu = YM / (2.0 * (1.0 + PR))

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            F_i = self.F[p_i]
            K_i = self.K[p_i0]

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
                    g = self.LgradW[p_i0, k]
                    s = ti.math.dot(g, xij0)
                    sumVs += self.ps.m_V0[p_k] * s
                    ti.atomic_add(self.ZE[p_k], self.ps.m_V0[p_k] * s * e_ij * K_i[j])

                ti.atomic_add(self.ZE[p_j],  e_ij * K_i[j])
                ti.atomic_add(self.ZE[p_i], -(1.0 + sumVs) * e_ij * K_i[j])

        for p in ti.grouped(self.ZE):
            self.ZE[p] *= alpha * mu


        # step 2 Mx + dt ** 2 * D^TKDx
        for p_i in ti.grouped(self.ps.x):

            # self.grad[p_i] = ti.math.vec3(0.0)

            if self.ps.material[p_i] != self.ps.material_solid:
                continue

            p_i0 = self.ps.cur2ori[p_i]
            # PL_i = self.P[p_i] @ self.L[p_i0]

            f_i = ti.math.vec3(0.0)

            for j in range(self.ps.solid_neighbors_num[p_i0]):
                p_j0 = self.ps.solid_neighbors[p_i0, j]
                p_j = self.ps.ori2cur[p_j0]

                xji0 = self.ps.x0[p_j] - self.ps.x0[p_i]
                PL_i = - self.P[p_i] @ self.LgradW[p_i0, j]
                PL_j = self.P[p_j] @ self.L[p_j0] @ self.gradW(xji0, self.ps.support_radius)

                f_i += self.ps.m_V0[p_j] * (PL_i + PL_j)
            Ax[p_i] = self.ps.m[p_i] * x[p_i] + self.ps.m_V0[p_i] * dt ** 2 * (f_i + self.ZE[p_i])



    def solve(self, alpha, YM, PR, dt):

        add(self.x_tmp, self.ps.x, dt, self.ps.v)
        self.compute_P(self.x_tmp, YM, PR)
        self.compute_ZE(alpha, YM, PR, self.x_tmp)
        self.compute_gradient(dt)

        # TODO: v +=  M^-1 * (grad)
        self.v_tmp.copy_from(self.ps.v)
        self.CG(x=self.a, b=self.grad, alpha=alpha, YM=YM, PR=PR, dt=dt)
        add(self.v_tmp, self.v_tmp, -1.0, self.a)
        self.ps.v.copy_from(self.v_tmp)


    @ti.kernel
    def update_surface_vertex(self):

        for k in ti.grouped(self.ps.x_s):
            x_k = ti.math.vec3(0.0)
            X_k = self.ps.x_0_s[k]
            s_k = self.ps.skinning_weight[k]
            for j in range(self.ps.surface_neighbor_num[k]):
                j0 = self.ps.surface_neighbor_idx[k, j]
                p_j = self.ps.ori2cur[j0]
                X_kj = X_k - self.ps.x0[j0]
                x_k += s_k * self.ps.m_V0[p_j] * (self.F[p_j] @ X_kj + self.ps.x[p_j]) * self.W(X_kj.norm(), self.ps.support_radius)

            self.ps.x_s[k] = x_k


    # Example usage: three vertex coord of ith triangle: tri_pos = obj["meshVertices"][obj["meshFaces"][i]]  # (3, 3)
    def apply_mesh_skinning(self):

        self.compute_F(self.ps.x)
        self.update_surface_vertex()

