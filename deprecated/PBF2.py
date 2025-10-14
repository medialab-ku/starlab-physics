import time
import numpy as np
from deprecated.sph_base import SPHBase
from math_utils import *



class PBF2Solver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)

        self.dt = self.ps.cfg.get_cfg("timeStepSize")

        self.nablaWij = self.spiky_kernel_derivative
        self.Wij = self.cubic_kernel
        self.lda = self.ps.pressure
        self.method = 1
        self.iisph = False
        self.num_substep = self.ps.cfg.get_cfg("numSubstepping")
        self.pcg_total_iter = 0

        self.adaptive_step_size = False
        self.use_pcg = True
        self.enable_DF = False
        self.pressure_boundary = False
        self.smooth_max = False
        self.density_error = False
        self.print_info = True
        
        self.print_pcg_iter  = False
        self.print_opt_iter  = True
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

        self.dp   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.c   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.p   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.k = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.Aii = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.Zii = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.dfdt = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.t = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.f = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Jx = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.test = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.var = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Hii   = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.Ap    = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.b_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.z_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.r_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.p_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)

        if self.ps.num_rigid_bodies > 0:
            self.I_rb   = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.mass_rb.shape)  # per rigid
            self.t_rb   = ti.Vector.field(n=3, dtype=float, shape=self.ps.mass_rb.shape)       # per rigid
            self.vsum_rb= ti.Vector.field(n=3, dtype=float, shape=self.ps.mass_rb.shape)       # per rigid
        else:
            self.I_rb = None
            self.t_rb = None
            self.vsum_rb = None

        self.matrix_type = 0

        # Iteration logging system
        self.enable_logging = False
        self.iteration_log = []  # Store [frame, matrix_type, iterations]
        self.current_frame = 0

        self.total_ratio = ti.field(dtype=float, shape=())
        self.active_particle_count = ti.field(dtype=float, shape=())

        self.use_max = False
        self.DF_tol = 0.1

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
    def compute_normal(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                self.ps.n[p_i] = ti.Vector.zero(float, self.ps.dim)
                continue
            n = ti.Vector.zero(float, self.ps.dim)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] != self.ps.material_fluid:
                    continue
                n += (self.ps.m[p_j] / (self.ps.density[p_j] + 1e-12)) * self.nablaWij(self.ps.x[p_i] - self.ps.x[p_j])
            self.ps.n[p_i] = self.ps.support_radius * n


    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        r   = x_i - x_j
        h   = self.ps.support_radius

        rn = r.norm()
        inv_rn = 1.0 / ti.max(rn, 1e-6 * h)
        r_hat  = r * inv_rn

        gradW = self.spiky_kernel_derivative(r)
        vxy   = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        eps2  = (0.01 * h) * (0.01 * h)

        # ---------- Cohesion & Viscosity ----------
        # Akinci2012
        # Akinci2013
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Cohesion
            f_coh = - self.surface_tension * self.ps.m[p_i] * self.ps.m[p_j] * self.cohesion_term(r) * r_hat
            # Curvature
            f_curv = - self.surface_tension * self.ps.m[p_i] * (self.ps.n[p_i] - self.ps.n[p_j])
            # Neighborhood deficiency correction K_ij
            K_ij = 2.0 * self.ps.density0[p_i] / (self.ps.density[p_i] + self.ps.density[p_j])
            # K_ij = ti.min(K_ij, 1.0)
            ret += K_ij * (f_coh + f_curv)

            # Viscosity (fluid-fluid)
            k_visc = 2.0 * (self.ps.dim + 2.0)
            f_v = k_visc * self.viscosity * (self.ps.m[p_j] / self.ps.density[p_j]) * (vxy / (rn * rn + eps2)) * gradW
            ret += K_ij * f_v

        # ---------- Adhesion ----------
        elif self.ps.material[p_j] == self.ps.material_solid:
            f_adh = - self.adhesion_coeff * self.ps.m[p_i] * self.ps.m[p_j] * self.adhesion_term(r) * r_hat
            ret += f_adh

            # (Optional) Boundary Viscosity
            boundary_viscosity = 0.0
            if boundary_viscosity > 0.0:
                k_visc = 2.0 * (self.ps.dim + 2.0)
                f_vb = k_visc * boundary_viscosity * (self.ps.m[p_j] / self.ps.density[p_i]) * (vxy / (rn * rn + eps2)) * gradW
                ret += f_vb
                if self.ps.is_dynamic_rigid_body(p_j):
                    # Two-way coupling
                    self.ps.acceleration[p_j] += -(f_vb) * self.ps.density0[p_i] / self.ps.density[p_j]

            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -(f_adh) * self.ps.density0[p_i] / self.ps.density[p_j]

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue

            acc = ti.Vector(self.g)
            self.ps.acceleration[p_i] = acc

            # Stabilize freshly emitted particles by skipping strong neighbor forces for a few substeps
            if self.ps.material[p_i] == self.ps.material_fluid and self.ps.age[p_i] >= 2:
                self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, acc)
                    # Write back the accumulated non-pressure forces into acceleration
                self.ps.acceleration[p_i] = acc

    @ti.kernel
    def advect_velocity(self, dt: float):
        # Symplectic Euler
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v_adv[p_i] = self.ps.v[p_i] + dt * self.ps.acceleration[p_i]
            else:
                self.ps.v_adv[p_i] = ti.math.vec3(0.0)


    @ti.kernel
    def advect_position(self, dt: float):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                # self.ps.v[p_i] += self.dt * self.ps.acceleration[p_i]
                self.ps.x[p_i] += dt * self.ps.v[p_i]

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

            # J_ii = ti.math.vec3(0.0)
            # denom_i = eps + self.Aii[p_i]
            # JtJ = ti.math.mat3(0.0)
            # if self.ps.density[p_i] <= self.ps.density0[p_i]:
            #     continue

            # for j in range(self.ps.fluid_neighbors_num[p_i]):
            #     p_j = self.ps.fluid_neighbors[p_i, j]
            #     grad_ij = self.ps.fluid_neighbors_values[p_i, j]
            #     J_ij = self.ps.m[p_j] * grad_ij
            #     self.ps.fluid_neighbors_JtJ[p_i, j] = Jij.outer_product(Jij)
                # if self.ps.is_dynamic[p_j]:
                #     self.Hii[p_j] += self.dfdt[p_j] * self.k[p_j] * JtJ_ij

                # J_ii -= J_ij
            # JtJ += J_ii.outer_product(J_ii)


    @ti.kernel
    def compute_Zii(self):
        eps = 1e-8
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                # self.Zii[p_i] = 0.5
                continue

            J_ii = ti.math.vec3(0.0)
            Gii = self.ps.m[p_i]*self.Aii[p_i]

            denom = Gii
            cnt = 0
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.is_dynamic_rigid_body(p_i) and (self.ps.object_id[p_j] == self.ps.object_id[p_i]):
                    continue
                if not self.ps.is_dynamic[p_j]:
                    continue
                cnt += 1
                grad_ij = self.ps.fluid_neighbors_values[p_i, j]
                J_ij = self.ps.m[p_j] * grad_ij
                J_ji = - self.ps.m[p_i] * grad_ij

                J_jj = ti.math.vec3(0.0)
                for ll in range(self.ps.fluid_neighbors_num[p_j]):
                    p_l = self.ps.fluid_neighbors[p_j, ll]
                    grad_jl = self.ps.fluid_neighbors_values[p_j, ll]
                    J_jl = self.ps.m[p_l] * grad_jl
                    J_jj -= J_jl

                Ji_dot_Jj = J_ii.dot(J_ji) + J_ij.dot(J_jj)
                denom += ti.abs(Ji_dot_Jj)
            
            if cnt > 0:
                denom /= cnt
            Zi = denom / (Gii + eps) 
            self.Zii[p_i] = ti.max(Zi, 1.0)


    @ti.kernel
    def compute_JtJ(self):

        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue
            self.Hii[p_i] = ti.math.mat3(0.0)
            # J_ii = ti.math.vec3(0.0)
            # denom_i = eps + self.Aii[p_i]
            JtJ = ti.math.mat3(0.0)
            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # grad_ij = self.ps.fluid_neighbors_values[p_i, j]
                JtJ_ij = self.ps.fluid_neighbors_JtJ[p_i, j]

                if self.ps.is_dynamic[p_j]:
                    self.Hii[p_j] += self.dfdt[p_j] * self.k[p_j] * JtJ_ij

                # J_ii -= J_ij
            # JtJ += J_ii.outer_product(J_ii)
            self.Hii[p_i] += self.dfdt[p_i] * self.k[p_i] * self.ps.fluid_neighbors_JtJ_ii[p_i]


    @ti.kernel
    def update_velocities(self, dt: float):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] = (self.ps.x[p_i] - self.ps.x_old[p_i]) / dt
            else:
                self.ps.v[p_i] = ti.math.vec3(0.0)


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
    def num_negative_p(self, p: ti.template()) -> int:
        cnt = 0
        for i in p:
            if p[i] > 0.0:
                cnt += 1
        return cnt  
    

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
                    ret[p_i] += (self.ps.m[p_j] * x[p_i] + self.ps.m[p_i] * x[p_j]) * self.ps.fluid_neighbors_values[p_i, j]
                else:
                    ret[p_i] += (self.ps.m[p_j] * x[p_i]) * self.ps.fluid_neighbors_values[p_i, j]


    @ti.kernel
    def compute_b(self, b: ti.template(), v: ti.template(), dt: float):

        dtSq = dt ** 2
        for p_i in ti.grouped(b):
            div_i = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.is_dynamic_rigid_body(p_i) and (self.ps.object_id[p_j] == self.ps.object_id[p_i]):
                    continue
                if self.ps.is_dynamic[p_j]:
                    div_i += self.ps.m[p_j] * (v[p_i] - v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    div_i += self.ps.m[p_j] * (v[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            b[p_i] = (self.ps.density[p_i] + dt * div_i - self.ps.density0[p_i]) / dtSq


    @ti.kernel
    def compute_b_div(self, b: ti.template(), v: ti.template(), dt: float):

        for p_i in ti.grouped(b):
            div_i = 0.0
            b[p_i] = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]

                if self.ps.is_dynamic[p_j]:
                    div_i += self.ps.m[p_j] * (v[p_i] - v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    div_i += self.ps.m[p_j] * (v[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            b[p_i] = div_i / dt


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
                    div_i += self.ps.m[p_j] * (self.ps.v[p_i] - self.ps.v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    div_i += self.ps.m[p_j] * (self.ps.v[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            avg_error = ti.max(self.ps.density[p_i] + dt * div_i - self.ps.density0[p_i], 0.0) / self.ps.density0[p_i]

        return avg_error


    @ti.kernel
    def measure_divergence(self, v: ti.template()) -> float:
        avg_error = 0.0
        for p_i in ti.grouped(v):
            div_i = 0.0
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    div_i += self.ps.m[p_j] * (self.ps.v[p_i] - self.ps.v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    div_i += self.ps.m[p_j] * (self.ps.v[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])
                # div_i += self.ps.m[p_j] * (v[p_i] - v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])

            avg_error += ti.max(div_i, 0.0) / (self.ps.density[p_i] + 1e-12)
            self.ps.divergence[p_i] = div_i


        avg_error /= self.ps.fluid_particle_num

        return avg_error

    def divergence_free_sovle_IISPH(self):

        self.compute_density(self.pressure_boundary)
        # self.precompute_values()
        self.ps.v.copy_from(self.ps.v_adv)
        self.v_tmp.copy_from(self.ps.v)
        self.compute_Aii(self.iisph_vanilla)
        # self.compute_b(self.b, self.ps.v, self.dt)
        self.compute_b_div(self.b, self.ps.v_adv, self.dt)

        tol = pow(10, -self.tol_opt + 1)
        self.p.fill(0.0)
        iter = 0
        for _ in range(self.max_iteration_opt):
            if self.iisph_vanilla:
                coef_wise_mul(self.y, self.Dii, self.p)
                self.compute_J_tr_x(self.tmp, self.y)
            else:
                self.compute_J_tr_x(self.tmp, self.p)

            # coef_wise_op(self.tmp, self.tmp, self.ps.m, 1)
            add(self.v_tmp, self.ps.v_adv, -self.dt, self.tmp)
            error = self.measure_divergence(self.v_tmp, self.dt)

            if error < tol and iter > 2:

                print(f" converged iter: {iter}. div: {error}")
                self.ps.v.copy_from(self.v_tmp)
                break

            iter += 1

            self.compute_J_x(self.Jx, self.tmp)
            add(self.r_jacobi, self.b, -1.0, self.Jx)

            coef_wise_op(self.dp, self.r_jacobi, self.Aii, 1)


            add(self.p, self.p, 0.3, self.dp)  # Initialize p with b
            max(self.p)  # Ensure non-negativity

        print(f"iter: {iter}. div: {error}")
        self.ps.v.copy_from(self.v_tmp)
        self.ps.v_adv.copy_from(self.ps.v)


    def constant_density_solve_IISPH(self):
        
        self.compute_density(self.pressure_boundary)
        # self.precompute_values()
        self.ps.v.copy_from(self.ps.v_adv)
        self.compute_Aii(self.pressure_boundary, self.iisph_vanilla)
        self.compute_b(self.b, self.ps.v, self.dt)

        tol = pow(10, -self.tol_opt)
        self.p.fill(0.0)
        iter = 0
        for _ in range(self.max_iteration_opt):
        
            # if self.iisph_vanilla:
            coef_wise_mul(self.y, self.Dii, self.p)
            self.compute_J_tr_x(self.pressure_boundary, self.tmp, self.y)
            # else:
            #     self.compute_J_tr_x(self.pressure_boundary, self.tmp, self.p)
                
                
            coef_wise_op(self.tmp, self.tmp, self.ps.m, 1)
            add(self.ps.v, self.ps.v_adv, -self.dt, self.tmp)

            self.rigid_compute_cm_and_vcm()
            self.rigid_compute_angular_velocity()
            self.rigid_project_velocities()

            error = self.measure_error(self.ps.v, self.dt)
            #
            # if error < tol and iter > 2:
            #
            #     print(f" converged iter: {iter}. error: {error}")
            #     break
            #
            iter += 1 

            self.compute_J_x(self.pressure_boundary, self.Jx, self.tmp)
            add(self.r_jacobi, self.b, -1.0, self.Jx)

            coef_wise_op(self.dp, self.r_jacobi, self.Aii, 1)
            add(self.p, self.p, self.omega, self.dp)  # Initialize p with b
            max(self.p)  # Ensure non-negativity
        # self.apply_rigid_pressure(self.dt)
        self.advect_position(self.dt)




    @ti.kernel
    def compute_gradient(self, grad: ti.template()):

        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue

            grad[p_i] = (self.dfdt[p_i] / self.Aii[p_i]) * grad[p_i]


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
    def compute_Ax2(self, Ax: ti.template(), Jx: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            if not self.ps.is_dynamic[p_i]:
                continue

            Ax[p_i] = 0.0

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                if self.ps.is_dynamic[p_j]:
                    Jx[p_i] += (x[p_i] - x[p_j]).dot(val_ij)
                else:
                    Jx[p_i] += x[p_i].dot(val_ij)

            Jx[p_i] *= self.dfdt[p_i] / self.Aii[p_i]


        for p_i in ti.grouped(x):
            if not self.ps.is_dynamic[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                Ax[p_i] += Jx[p_i] * val_ij

                if self.ps.is_dynamic[p_j]:
                    Ax[p_j] -= Jx[p_i] * val_ij

    @ti.kernel
    def apply_precondition(self, dx: ti.template(), Hii: ti.template(), grad: ti.template()):

        for p_i in ti.grouped(self.ps.x):

            dx[p_i] = ti.math.vec3(0.0)
            I3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            if not self.ps.is_dynamic[p_i]:
                continue

            if self.precondition == 1:
                H = self.ps.m[p_i] * I3x3 + Hii[p_i]
                dx[p_i] = H.inverse() @ grad[p_i]

            elif self.precondition == 2:
                dx[p_i] = (self.ps.m_inv[p_i]* I3x3) @ grad[p_i]

            elif self.precondition == 3:
                dx[p_i] = grad[p_i]


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

    @ti.kernel
    def compute_gradient(self, grad: ti.template()):


        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue

            grad[p_i] += self.ps.m[p_i] * (self.dx[p_i] - self.s[p_i])


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
        if self.smooth_max and r_old > 1e-12:
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

                # Collect PCG residual error per iteration
                self.stats_pcg_error.append(float(r_new))
                try:
                    self.stats_pcg_error_frame.append(int(self.current_frame))
                except Exception:
                    self.stats_pcg_error_frame.append(0)
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

    def Jacobi(self):

        self.p.fill(0.0)
        self.dx.copy_from(self.s)

        iter = 0

        tol = pow(10, -self.tol_opt)

        if not self.use_max:
            self.compute_f(self.f, self.c, self.eps)
            self.compute_f_derivative(self.dfdt, self.c, self.eps)


        for _ in range(self.max_iteration_opt):

            self.compute_J_x(self.pressure_boundary, self.Jx, self.dx)

            if self.use_max:
                add(self.r_jacobi, self.Jx, 1.0, self.f)

            else:
                coef_wise_op(self.Jx, self.Jx, self.dfdt, 0)
                add(self.r_jacobi, self.Jx, 1.0, self.f)

            err = self.measure_error2(self.r_jacobi)
            # print(err)
            if (err < tol and iter > 2) or iter >= self.max_iteration_opt:
                # if self.print_info:
                #     print(f"CD iter: {iter}, err: {err}")
                break

            coef_wise_op(self.dp, self.r_jacobi, self.Aii, 1)
            add(self.p, self.p, self.omega, self.dp)

            if self.use_max:
                max(self.p)
            else:
                coef_wise_op(self.p, self.p, self.dfdt, 0)


            self.compute_J_tr_x(self.pressure_boundary, self.tmp, self.p)
            self.compute_inv_M_x(self.tmp, self.tmp)
            add(self.dx, self.s, -1.0, self.tmp)

            iter += 1

        if self.print_info:
            print(f"CD iter: {iter}")

    @ti.kernel
    def compute_constraint(self):

        for p_i in ti.grouped(self.ps.x):

            self.c[p_i] = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue

            self.c[p_i] = (self.ps.density[p_i] - self.ps.density0[p_i])

    @ti.kernel
    def compute_d(self):

        for p_i in ti.grouped(self.ps.x):

            self.d_v[p_i] = ti.math.vec3(0.0)
            self.d_l[p_i] = 0.0

            if not self.ps.is_dynamic[p_i]:
                continue

            self.d_v[p_i] = self.ps.m[p_i] * self.s[p_i]
            self.d_l[p_i] = -self.c[p_i]


    @ti.kernel
    def compute_invC_x(self, ret_v: ti.template(), ret_l: ti.template(), x_v: ti.template(), x_l: ti.template()):

        # invC x_v =
        # step 1
        for p_i in ti.grouped(self.ps.x):

            ret_v[p_i] = ti.math.vec3(0.0)
            ret_l[p_i] = 0.0
            self.tmp[p_i] = ti.math.vec3(0.0)

            if not self.ps.is_dynamic[p_i]:
                continue

            ret_v[p_i] = x_v[p_i]
            self.tmp[p_i] = x_v[p_i] * self.ps.m_inv[p_i]

        for p_i in ti.grouped(self.ps.x):

            if not self.ps.is_dynamic[p_i]:
                continue

            ret_l[p_i] = x_l[p_i]

            if self.dfdt[p_i] > 0.0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    # val = ti.cast(self.ps.material[p_i], float)
                    if self.ps.is_dynamic[p_i]:
                        if self.ps.is_dynamic[p_j]:
                            ret_l[p_i] -=self.ps.m[p_j] * (self.tmp[p_i] - self.tmp[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                        else:
                            ret_l[p_i] -= (self.ps.m[p_j] * self.tmp[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

        # step 2
        for p_i in ti.grouped(self.ps.x):

            # ret_v[p_i] = ti.math.vec3(0.0)
            # ret_l[p_i] = 0.0

            if not self.ps.is_dynamic[p_i]:
                continue

            ret_v[p_i] = ret_v[p_i] * self.ps.m_inv[p_i]
            ret_l[p_i] = - 0.5 * (self.dfdt[p_i] / self.Aii[p_i]) * ret_l[p_i]

        # step 3
        for p_i in ti.grouped(self.ps.x):

            self.tmp[p_i] = ti.math.vec3(0.0)
            if not self.ps.is_dynamic[p_i]:
                continue

            # num_f += 1

            if self.dfdt[p_i] > 0.0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    # val = ti.cast(self.ps.material[p_i], float)
                    if self.ps.is_dynamic[p_i]:
                        if self.ps.is_dynamic[p_j]:
                            self.tmp[p_i] += (self.ps.m[p_j] * ret_l[p_i] + self.ps.m[p_i] * ret_l[p_j]) * self.ps.fluid_neighbors_values[p_i, j]
                        else:
                            self.tmp[p_i] += (self.ps.m[p_j] * ret_l[p_i]) * self.ps.fluid_neighbors_values[p_i, j]


            ret_v[p_i] = ret_v[p_i] - self.tmp[p_i] * self.ps.m_inv[p_i]

    @ti.kernel
    def compute_B_x(self, ret_v: ti.template(), ret_l: ti.template(), x_v: ti.template(), x_l: ti.template()):

        #B    v   = Mv + J^tr * lambda
        #B lambda = Jv
        for p_i in ti.grouped(self.ps.x):

            ret_v[p_i] = ti.math.vec3(0.0)
            ret_l[p_i] = 0.0

            if not self.ps.is_dynamic[p_i]:
                continue

            ret_v[p_i] = self.ps.m[p_i] * x_v[p_i]

            if self.dfdt[p_i] > 0.0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    if self.ps.is_dynamic[p_i]:
                        if self.ps.is_dynamic[p_j]:
                            ret_v[p_i] += (self.ps.m[p_j] * x_l[p_i] + self.ps.m[p_i] * x_l[p_j]) * self.ps.fluid_neighbors_values[p_i, j]
                        else:
                            ret_v[p_i] += (self.ps.m[p_j] * x_l[p_i]) * self.ps.fluid_neighbors_values[p_i, j]

                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    if self.ps.is_dynamic[p_j]:
                        ret_l[p_i] += self.ps.m[p_j] * (x_v[p_i] - x_v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                    else:
                        ret_l[p_i] += self.ps.m[p_j] * (x_v[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

    @ti.kernel
    def evaluateConstraints(self, x_v: ti.template(), x_l: ti.template()) -> int:

        flag = 0
        for p_i in ti.grouped(self.ps.x):

            self.Jx[p_i] = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.is_dynamic[p_j]:
                    self.Jx[p_i] += self.ps.m[p_j] * (x_v[p_i] - x_v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    self.Jx[p_i] += self.ps.m[p_j] * (x_v[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            a = self.Jx[p_i] + self.c[p_i]
            if self.dfdt[p_i] > 0.0:
                if a <= 0 and x_l[p_i] <= 0:
                    x_l[p_i] = 0.0
                    self.dfdt[p_i] = 0.0
                    flag = 1

            elif a >= 0.0:
                # print("test")
                self.dfdt[p_i] = 1.0
                flag = 1
        # if flag > 0:
        #     print("changed")

        return flag


    @ti.kernel
    def compute_avg_density_error(self, a: ti.template()) -> float:

        value = 0.0
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue

            value += a[p_i] / self.ps.density0[p_i]
        value /= self.ps.dynamic_particle_num

        return value

    def constant_density_solve(self):
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
            if self.iisph:
                self.f.copy_from(self.t)
                max(self.f)
                err = self.compute_avg_density_error(self.f)

                err_log = self.dot(p, p)
                # Collect optimizer error per iteration
                self.stats_opt_error.append(float(err_log))
                try:
                    self.stats_opt_error_frame.append(int(self.current_frame))
                except Exception:
                    self.stats_opt_error_frame.append(0)
                if self.print_opt_error:
                    print(f"opt error: {err_log}")

                # Count this iteration before break check so that opt_iter matches logged errors
                opt_iter += 1
                if ((err < pow(10, -self.tol_opt)) and (opt_iter > 1)) or (opt_iter >= self.max_iteration_opt):
                    break

            

                coef_wise_div(self.dp, self.t, self.Aii)
                self.add(self.p, self.p, 0.5, self.dp)
                max(self.p)
                self.compute_J_tr_x(g, self.p)
                coef_wise_div(g, g, self.ps.m)
                self.add(d, d, -1.0, g)

            else:
                err_log = 0.0
                self.compute_f(self.f, self.t, self.eps)

                # self.f.copy_from(self.t)
                # max(self.f)
                self.compute_f_derivative(self.dfdt, self.t, self.eps)
                coef_wise_mul(self.f, self.f, self.k)
                self.compute_J_tr_x(g, self.f)
                # coef_wise_div(p, g, self.ps.m)

                # if self.smooth_max:

                #     if not self.use_pcg:
                p.fill(0.0)

                self.PCG(x=p, b=g)

                #Bender et al. 2014 (Constant density solver OF DFSPH)
                # else:
                #     coef_wise_div(p, g, self.ps.m)

                self.add(d, d, -self.omega, p)
                # err = inf_norm(p)
                # err = dot2(Jd, Jd)
                if self.density_error:
                    err = self.compute_avg_density_error(self.f)
                else:
                    err = self.dot(p, p)
                err_log = self.dot(p, p)


                # Collect optimizer error per iteration
                self.stats_opt_error.append(float(err_log))
                try:
                    self.stats_opt_error_frame.append(int(self.current_frame))
                except Exception:
                    self.stats_opt_error_frame.append(0)
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



        #x_n+1
        self.add(self.ps.x, self.ps.x, 1.0, self.dx)
        # self.enforce_boundary_3D(self.ps.material_fluid)

        #v_n+1_tmp
        self.update_velocities(self.dt)
        self.ps.x.copy_from(self.ps.x_old)

        self.v_tmp.copy_from(self.ps.v)
        coef_wise_mul(self.v_tmp, self.v_tmp, self.ps.m)
        kinetic_energy = 0.5 * self.dot(self.ps.v, self.v_tmp)
        if self.print_kinetic_energy:
            print(f"kinetic energy: {kinetic_energy}")
        self.stats_kinetic_energy.append(float(kinetic_energy))

        if self.ps.num_rigid_bodies > 0:
            self.rigid_compute_cm_and_vcm()
            self.rigid_compute_angular_velocity()
            self.rigid_project_velocities()

    def divergence_free_solve(self):
        self.enforce_boundary_3D(self.ps.material_fluid)
        self.ps.initialize_particle_system()
        self.ps.search_neighbours(self.ps.x)
        self.compute_density(self.pressure_boundary)
        self.compute_constraint(self.pressure_boundary, False)
        self.compute_Aii(self.pressure_boundary, False)


        self.v_tmp.copy_from(self.ps.v)
        Jv = self.Jx
        iter = 0

        print("-------------")
        self.compute_J_x(self.pressure_boundary, Jv, self.v_tmp)
        self.compute_f_derivative(self.dfdt, self.c, self.eps)
        self.p.fill(0.0)
        for _ in range(self.max_iteration_opt):

            self.p.fill(0.0)
            self.compute_J_x(self.pressure_boundary, Jv, self.ps.v)
            coef_wise_op(Jv, Jv, self.dfdt, 0)
            err = self.measure_error2(Jv)
            if (err < self.DF_tol and iter > 2) or iter == self.max_iteration_opt:
                if self.print_info:
                    print(f"DF iter: {iter}, err: {err}")

                break

            coef_wise_op(self.dp, Jv, self.Aii, 1)
            add(self.p, self.p, self.omega, self.dp)
            coef_wise_op(self.p, self.p, self.dfdt, 0)
            max(self.p)


            self.compute_J_tr_x(self.pressure_boundary, self.tmp, self.p)
            self.compute_inv_M_x(self.tmp, self.tmp)
            add(self.ps.v, self.ps.v, -1.0, self.tmp)

            iter += 1

    def compute_cfl_dt(self, dt_in: float) -> float:
        v_max = inf_norm(self.ps.v)
        dt_upper_bound = 0.4 * (self.ps.particle_diameter / (v_max + 1e-12))
        dt_lower_bound = 0.001
        return max(dt_lower_bound, min(dt_upper_bound, float(dt_in)))

    @ti.kernel
    def compute_variance(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                self.var[p_i] = 0.0
                continue
            sum_i = 0.0
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] != self.ps.material_fluid:
                    continue
                rho_ij = self.ps.density[p_i] - self.ps.density[p_j]
                # if rho_ij >= 1e-6:
                #     print('test')
                xij = (self.ps.x[p_i] - self.ps.x[p_j]).norm()
                # if xij > 1e-6:
                #     term = rho_ij / xij
                term = rho_ij
                sum_i += term * term
            self.var[p_i] = ti.sqrt(sum_i)
    @ti.kernel
    def advect_position(self, dt: float):
         for p_i in ti.grouped(self.ps.x):
            # if self.ps.material[p_i] == self.ps.material_fluid:
            self.ps.x[p_i] = self.ps.x_old[p_i] + dt * self.ps.v[p_i]   
    def substep(self):

        self.ps.initialize_particle_system()
        self.ps.search_neighbours(self.ps.x)

        dt_original = self.dt

        if self.cfl:
            v_max = inf_norm(self.ps.v)
            dt_upper_bound = 0.4 * (self.ps.particle_diameter / (v_max + 1e-12))
            dt_lower_bound = 0.001
            self.dt = ti.max(dt_lower_bound, ti.min(dt_upper_bound, self.dt))
            print(f"use smaller time step: {self.dt}")
        
        self.pcg_total_iter = 0
        self.compute_normal()
        self.compute_non_pressure_forces()
        self.advect_velocity(self.dt)
        self.constant_density_solve()
        
        
        self.advect_position(self.dt)

        # if self.enable_DF:
        #     self.divergence_free_solve()

        # error = self.measure_divergence(self.ps.v)
        # self.measure_divergence(self.ps.v, self.dt)
        self.dt = dt_original  # Reset dt to original value after substep