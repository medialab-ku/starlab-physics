import time
import numpy as np
from math_utils import *

@ti.data_oriented
class Pressure:
    def __init__(self, particle_system):
        # super().__init__(particle_system)

        self.ps = particle_system
        self.dt = self.ps.cfg.get_cfg("timeStepSize")
        self.lda = self.ps.pressure
        self.time = 0.0
        self.nablaWij = self.spiky_kernel_derivative
        self.Wij = self.cubic_kernel

        self.num_substep = self.ps.cfg.get_cfg("numSubstepping")
        self.pcg_total_iter = 0

        self.adaptive_step_size = False
        self.use_pcg = True
        self.enable_DF = False
        self.pressure_boundary = False
        self.smooth_max = False
        self.density_error = False
        self.print_info = True

        self.print_pcg_iter = False
        self.print_opt_iter = True
        self.print_pcg_error = False
        self.print_opt_error = False
        self.print_elapsed_time = False
        self.print_kinetic_energy = False

        self.volume_constraint = False
        self.tol_opt = 4
        self.omega = 1.0
        self.eps = 0.001
        self.cfl = False
        self.max_iteration_opt = 1000
        self.max_iteration_pcg = 1000
        self.tol_pcg = 10

        # Stats containers
        # These are plain Python lists to minimize Taichi interaction overhead.
        self.stats_elapsed_ms = []
        # detailed elapsed time

        self.stats_opt_iter = []
        self.stats_opt_error = []
        self.stats_pcg_iter = []
        self.stats_pcg_error = []
        self.stats_kinetic_energy = []
        # Per-iteration → timestep mapping (global substep index)
        self.stats_opt_error_frame = []
        self.stats_pcg_error_frame = []

        # Taichi fields and buffers
        self.tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.grad = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.dx = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.a = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.error = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.s = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)

        self.dp = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.c = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.p = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.k = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.Aii = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.Zii = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.dfdt = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.t = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.f = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Jx = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.test = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.var = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Hii = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.Ap = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.b_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.z_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.r_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.p_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)



    # -----------------------------
    # Stats helpers
    # -----------------------------
    def clear_stats(self):
        self.stats_elapsed_ms.clear()
        self.stats_opt_iter.clear()
        self.stats_opt_error.clear()
        self.stats_pcg_iter.clear()
        self.stats_pcg_error.clear()
        self.stats_opt_error_frame.clear()
        self.stats_pcg_error_frame.clear()
        self.stats_kinetic_energy.clear()

    def get_stats_numpy(self):
        """Return stats as numpy arrays. Keys:
        - elapsed_time_ms
        - opt_iter
        - opt_error
        - opt_error_frame
        - pcg_iter
        - pcg_error
        - pcg_error_frame
        """
        return {
            "elapsed_time_ms": np.asarray(self.stats_elapsed_ms, dtype=np.float64),
            "opt_iter": np.asarray(self.stats_opt_iter, dtype=np.int32),
            "opt_error": np.asarray(self.stats_opt_error, dtype=np.float64),
            "opt_error_frame": np.asarray(self.stats_opt_error_frame, dtype=np.int32),
            "pcg_iter": np.asarray(self.stats_pcg_iter, dtype=np.int32),
            "pcg_error": np.asarray(self.stats_pcg_error, dtype=np.float64),
            "pcg_error_frame": np.asarray(self.stats_pcg_error_frame, dtype=np.int32),
            "kinetic_energy": np.asarray(self.stats_kinetic_energy, dtype=np.float64)
        }


    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

    @ti.func
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def spiky_kernel_derivative(self, r):
        h = self.ps.support_radius
        k = 1.0
        if self.ps.dim == 1:
            k = 15 / 4
        elif self.ps.dim == 2:
            k = 30 / (np.pi * h ** 3)
        elif self.ps.dim == 3:
            k = 45 / (np.pi * h ** 6)

        r_norm = r.norm()

        if r_norm < 1e-6:
            r_norm = 1e-6
        # res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        # if 1e-5 < r_norm < h:
        grad_q = r / r_norm
        res = -k * ((h - r_norm) ** 2) * grad_q
        return res

    @ti.kernel
    def precompute_values(self):

        for p_i in ti.grouped(self.ps.x):
            x_i = self.ps.x[p_i]
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.fluid_neighbors_values[p_i, j] = self.nablaWij(x_i - x_j)

    @ti.kernel
    def compute_density(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue

            self.ps.density[p_i] = self.ps.m[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            x_i = self.ps.x[p_i]
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # Fluid neighbors
                x_j = self.ps.x[p_j]
                self.ps.fluid_neighbors_values[p_i, j] = self.nablaWij(x_i - x_j)
                den += self.ps.m[p_j] * self.cubic_kernel((x_i - x_j).norm())

            # self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den


    @ti.kernel
    def compute_Aii(self):

        eps = 1e-3
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue

            Aii = 0.0
            J_ii = ti.math.vec3(0.0)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]

                J_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                self.ps.fluid_neighbors_JtJ[p_i, j] = J_ij.outer_product(J_ij)

                # if self.ps.is_dynamic_rigid_body(p_i) and (self.ps.object_id[p_j] == self.ps.object_id[p_i]):
                #     continue
                if self.ps.is_dynamic[p_j]:
                    Aii += J_ij.dot(J_ij) * self.ps.m_inv[p_j]

                J_ii -= J_ij
            Aii += J_ii.dot(J_ii) * self.ps.m_inv[p_i]

            self.ps.fluid_neighbors_JtJ_ii[p_i] = J_ii.outer_product(J_ii)
            self.Aii[p_i] = Aii + eps
            self.k[p_i] = 1.0 / (Aii + eps)



    @ti.kernel
    def compute_J_x(self, ret: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            ret_i = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # if self.ps.is_dynamic_rigid_body(p_i) and (self.ps.object_id[p_j] == self.ps.object_id[p_i]):
                #     continue

                if self.ps.is_dynamic[p_j]:
                    ret_i += self.ps.m[p_j] * (x[p_i] - x[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    ret_i += self.ps.m[p_j] * (x[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            ret[p_i] = ret_i

    @ti.kernel
    def compute_J_tr_x(self, ret: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            ret[p_i] = ti.math.vec3(0.0)
            if not self.ps.is_dynamic[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # if self.ps.is_dynamic_rigid_body(p_i) and (self.ps.object_id[p_j] == self.ps.object_id[p_i]):
                #     continue

                if self.ps.is_dynamic[p_j]:
                    ret[p_i] += (self.ps.m[p_j] * x[p_i] + self.ps.m[p_i] * x[p_j]) * self.ps.fluid_neighbors_values[
                        p_i, j]
                else:
                    ret[p_i] += (self.ps.m[p_j] * x[p_i]) * self.ps.fluid_neighbors_values[p_i, j]


    @ti.kernel
    def measure_error(self, v: ti.template(), dt: float) -> float:

        avg_error = 0.0
        for p_i in ti.grouped(v):
            div_i = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.is_dynamic[p_j]:
                    div_i += self.ps.m[p_j] * (self.ps.v[p_i] - self.ps.v[p_j]).dot(
                        self.ps.fluid_neighbors_values[p_i, j])
                else:
                    div_i += self.ps.m[p_j] * (self.ps.v[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            avg_error = ti.max(self.ps.density[p_i] + dt * div_i - self.ps.density0[p_i], 0.0) / self.ps.density0[p_i]

        return avg_error



    @ti.kernel
    def compute_Ax(self, Ax: ti.template(), Jx: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            Jx[p_i] = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue

            if self.dfdt[p_i] >= 0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                    if self.ps.is_dynamic[p_j]:
                        Jx[p_i] += (x[p_i] - x[p_j]).dot(val_ij)
                    else:
                        Jx[p_i] += x[p_i].dot(val_ij)

            Jx[p_i] *= (self.dfdt[p_i] * self.k[p_i])

        for p_i in ti.grouped(x):
            if not self.ps.is_dynamic[p_i]:
                continue

            Ax[p_i] = self.ps.m[p_i] * x[p_i]

        for p_i in ti.grouped(x):
            if not self.ps.is_dynamic[p_i]:
                continue

            if self.dfdt[p_i] >= 0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                    Ax[p_i] += Jx[p_i] * val_ij

                    if self.ps.is_dynamic[p_j]:
                        Ax[p_j] -= Jx[p_i] * val_ij


    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> float:

        ret = 0.0
        for p_i in ti.grouped(self.ps.x):

            if not self.ps.is_dynamic[p_i]:
                continue

            ret += ti.math.dot(a[p_i], b[p_i])

        return ret

    @ti.kernel
    def add(self, ret: ti.template(), v0: ti.template(), scale: float, v1: ti.template()):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue

            ret[p_i] = v0[p_i] + scale * v1[p_i]

    def PCG(self, x, b):

        r = self.r_pcg
        p = self.p_pcg
        Ap = self.Ap
        Jp = self.dp

        self.compute_Ax(Ap, Jp, x)
        self.add(r, b, -1.0, Ap)

        r_old = self.dot(r, r)
        pcg_iter = 0
        # if r_old > pow(10, -self.tol_pcg):
        if r_old > 1e-12:
            p.copy_from(r)
            for _ in range(self.max_iteration_pcg):
                # Ap.fill(0.0)
                self.compute_Ax(Ap, Jp, p)

                pAp = self.dot(p, Ap)
                if pAp < 0.0:
                    print("Warning: non-positive definite matrix!")

                alpha = r_old / pAp

                self.add(x, x, alpha, p)
                self.add(r, r, -alpha, Ap)

                pcg_iter += 1
                r_new = self.dot(r, r)
                if self.print_pcg_error:
                    print(f"PCG error: {r_new}")

                # z.copy_from(r)
                if r_new < pow(10, -self.tol_pcg) or pcg_iter >= self.max_iteration_pcg:
                    break

                # rz_new = self.dot(r, z)
                beta = r_new / r_old
                self.add(p, r, beta, p)
                r_old = r_new

        # Collect PCG iteration count per PCG solve
        self.pcg_total_iter += pcg_iter

    @ti.kernel
    def compute_f(self, ret: ti.template(), x: ti.template(), eps: float):

        for p_i in ti.grouped(x):

            ret[p_i] = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue

            if x[p_i] >= 0.0:
                a_i = eps * self.ps.density0[p_i]
                ret[p_i] = ti.sqrt(ti.pow(x[p_i], 2) + a_i ** 2) - a_i
                # ret[p_i] = x[p_i]

    @ti.kernel
    def compute_f_derivative(self, ret: ti.template(), x: ti.template(), eps: float):

        # a = 0
        for p_i in ti.grouped(self.ps.x):
            ret[p_i] = 0.0

            if not self.ps.is_dynamic[p_i]:
                continue

            if x[p_i] >= 0.0:
                a_i = eps * self.ps.density0[p_i]
                ret[p_i] = x[p_i] / ti.sqrt(ti.pow(x[p_i], 2) + a_i ** 2)
                # ret[p_i] = 1.0


    @ti.kernel
    def compute_constraint(self):

        for p_i in ti.grouped(self.ps.x):

            self.c[p_i] = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue

            self.c[p_i] = (self.ps.density[p_i] - self.ps.density0[p_i])


    @ti.kernel
    def compute_avg_density_error(self, a: ti.template()) -> float:

        value = 0.0
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue

            value += a[p_i] / self.ps.density0[p_i]
        value /= self.ps.dynamic_particle_num

        return value

    @ti.kernel
    def update_velocities(self, dt: float):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] = (self.ps.x[p_i] - self.ps.x_old[p_i]) / dt
            else:
                self.ps.v[p_i] = ti.math.vec3(0.0)


    def solve(self):
        t_start = time.perf_counter()
        self.ps.x_old.copy_from(self.ps.x)
        add(self.ps.y, self.ps.x, self.dt, self.ps.v_adv)

        self.ps.x.copy_from(self.ps.y)

        self.compute_density()
        self.compute_constraint()
        self.compute_Aii()

        add(self.s, self.ps.y, -1.0, self.ps.x)
        Jd = self.Jx
        d = self.dx
        d.copy_from(self.s)
        # d.fill(0.0)
        c = self.c
        g = self.tmp
        # P = self.Hii
        p = self.a

        opt_iter = 0
        for _ in range(self.max_iteration_opt):

            self.compute_J_x(Jd, d)
            self.add(self.t, Jd, 1.0, c)

            err_log = 0.0
            self.compute_f(self.f, self.t, self.eps)
            self.compute_f_derivative(self.dfdt, self.t, self.eps)
            coef_wise_mul(self.f, self.f, self.k)
            self.compute_J_tr_x(g, self.f)

            p.fill(0.0)

            self.PCG(x=p, b=g)
            self.add(d, d, -self.omega, p)

            if self.density_error:
                err = self.compute_avg_density_error(self.f)
            else:
                err = self.dot(p, p)
            err_log = self.dot(p, p)


            self.stats_opt_error.append(float(err_log))
            if self.print_opt_error:
                print(f"opt error: {err_log}")

            # Count this iteration before break check so that opt_iter matches logged errors
            opt_iter += 1
            if ((err < pow(10, -self.tol_opt)) and (opt_iter > 2)) or (opt_iter >= self.max_iteration_opt):
                break

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        # Collect elapsed time per outer solve
        self.stats_elapsed_ms.append(float(elapsed_ms))

        if self.print_elapsed_time:
            print(f"elapsed time: {elapsed_ms:.3f} ms")

        self.stats_pcg_iter.append(int(self.pcg_total_iter))
        if self.print_pcg_iter:
            print(f"PCG iter: {int(self.pcg_total_iter)}")

        # Collect optimizer iteration count per outer solve
        self.stats_opt_iter.append(int(opt_iter))
        if self.print_opt_iter:
            print(f"opt iter: {opt_iter}")

        # x_n+1
        self.add(self.ps.x, self.ps.x, 1.0, self.dx)
        # self.enforce_boundary_3D(self.ps.material_fluid)

        # v_n+1_tmp
        self.update_velocities(self.dt)
        self.ps.x.copy_from(self.ps.x_old)
