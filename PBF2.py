import taichi as ti
import math
import os
import json
from sph_base import SPHBase
from math_utils import *



class PBF2Solver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)

        self.surface_tension = 0.001
        self.dt = self.ps.cfg.get_cfg("timeStepSize")

        self.nablaWij = self.spiky_kernel_derivative
        self.Wij = self.cubic_kernel
        self.lda = self.ps.pressure
        self.method = 1
        self.iisph_vanilla = False 
        self.num_substep = self.ps.cfg.get_cfg("numSubstepping")

        self.adaptive_step_size = False
        self.use_pcg = True
        self.enable_DF = False
        self.pressure_boundary = False
        self.smooth_max = False
        self.print_info = True
        self.print_pcg_iter  = False
        self.print_opt_iter  = False
        self.print_pcg_error = False
        self.print_opt_error = False
        self.print_elapsed_time = True
        self.volume_constraint = False
        self.tol_opt = 2
        self.omega = 0.5
        self.eps = 0.01
        self.cfl = False
        self.max_iteration_opt = 1000
        self.max_iteration_pcg = 1000
        self.tol_pcg = 3

        self.tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.grad = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.dx = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.a = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.error = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.s = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.x_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.dp   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.c   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.p   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.k = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.Aii = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.dfdt = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.f = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.t = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Jx = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.test = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Hii   = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.Ap    = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.b_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.z_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.r_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.p_pcg = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)


        self.I_rb   = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.mass_rb.shape)  # per rigid
        self.t_rb   = ti.Vector.field(n=3, dtype=float, shape=self.ps.mass_rb.shape)       # per rigid
        self.vsum_rb= ti.Vector.field(n=3, dtype=float, shape=self.ps.mass_rb.shape)       # per rigid

        self.stats_iter = 0
        self.stats_pcg_iter = 0

        self.matrix_type = 0
        
        # Iteration logging system
        self.enable_logging = False
        self.iteration_log = []  # Store [frame, matrix_type, iterations]
        self.current_frame = 0

        self.total_ratio = ti.field(dtype=float, shape=())
        self.active_particle_count = ti.field(dtype=float, shape=())

        self.use_max = False
        self.DF_tol = 0.1

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

        cnt = 0
        total_static_neighbors = 0
        num_dynamic_rigid = 0
        # print(self.ps.m[0] * self.cubic_kernel(0.0))
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



        avg_static_neighbors = 0.0
        if num_dynamic_rigid > 0:
            avg_static_neighbors = total_static_neighbors / num_dynamic_rigid
        # print(f"num high density rigid body: {cnt}, avg static neighbors per dynamic rigid: {avg_static_neighbors}")




                J_ij = Dii * self.ps.m[p_j] * grad_ij
                    self.Hii[p_i] += J_ij.outer_product(J_ij) / denom_i
            self.Hii[p_i] += J_ii.outer_product(J_ii) / denom_i
        ret /= self.ps.fluid_particle_num

    @ti.kernel
    def update_variables_pbf(self):

        ret = 0.0
        eps = 1e-3
        I3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for p_i in ti.grouped(self.ps.x):

            self.c[p_i] = 0.0
            self.p[p_i] = 0.0
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            self.c[p_i] = self.ps.density[p_i] - self.ps.density0[p_i]

            if self.c[p_i] <= 0.0:
                self.dfdt[p_i] = 0.0
            else:
                self.dfdt[p_i] = 1.0

            self.p[p_i] = (self.dfdt[p_i] / self.Aii[p_i]) * ti.max(self.c[p_i], 0.0)

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]

        ############## Surface Tension ###############
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(
                    ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())

        ############### Viscosoty Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)

        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
                    r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(r)
            ret += f_v
        elif self.ps.material[p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = d * boundary_viscosity * (self.ps.m[p_j] / (self.ps.density[p_i])) * v_xy / (
                    r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(r)
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]

    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            # d_v = ti.Vector([0.0, 0.0, 0.0])
            self.ps.acceleration[p_i] = d_v
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[p_i] = d_v

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
                if self.ps.is_dynamic_rigid_body(p_i) and (self.ps.object_id[p_j] == self.ps.object_id[p_i]):
                    continue
                if self.ps.is_dynamic[p_j]:
                    Aii += J_ij.dot(J_ij) / self.ps.m[p_j]

                J_ii -= J_ij
            Aii += J_ii.dot(J_ii) / self.ps.m[p_i]
            self.Aii[p_i] = Aii + eps
            self.k[p_i] = 1.0 / (Aii + eps)


    @ti.kernel
    def compute_JtJ(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue


            self.Hii[p_i] = ti.math.mat3(0.0)
            J_ii = ti.math.vec3(0.0)
            # denom_i = eps + self.Aii[p_i]
            JtJ = ti.math.mat3(0.0)
            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                grad_ij = self.ps.fluid_neighbors_values[p_i, j]
                J_ij = self.ps.m[p_j] * grad_ij

                if self.ps.material[p_j] == self.ps.material_fluid:
                    self.Hii[p_j] += self.dfdt[p_j] * self.k[p_j] * J_ij.outer_product(J_ij)

                J_ii -= J_ij
            JtJ += J_ii.outer_product(J_ii)
            self.Hii[p_i] += self.dfdt[p_i] * self.k[p_i] * JtJ

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
                if self.ps.is_dynamic_rigid_body(p_i) and (self.ps.object_id[p_j] == self.ps.object_id[p_i]):
                    continue

                if self.ps.is_dynamic[p_j]:
                    ret_i += self.ps.m[p_j] * (x[p_i] - x[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    ret_i += self.ps.m[p_j] * (x[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            ret[p_i] = ret_i

    @ti.kernel
    def compute_J_x_active(self, ret: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            ret[p_i] = 0.0

            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    ret[p_i] += self.ps.m[p_j] * (x[p_i] - x[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    ret[p_i] += self.ps.m[p_j] * (x[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])


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
                if self.ps.is_dynamic_rigid_body(p_i) and (self.ps.object_id[p_j] == self.ps.object_id[p_i]):
                    continue

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
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]

                if self.ps.material[p_j] == self.ps.material_fluid:
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


    def pressure_solve(self):
        
        if self.method == 0:
            self.constant_density_solve_IISPH()



    @ti.kernel
    def clamp_velocity(self, v: ti.template(), vmax: float):
        for p_i in ti.grouped(v):
            vi = v[p_i]
            n = vi.norm()
            if n > vmax:
                v[p_i] = (vmax / (n + 1e-12)) * vi



    @ti.kernel
    def compute_invM_J_tr_D_x(self, ret: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            ret_i = ti.math.vec3(0.0)

            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            inv_density_2 = 1.0 / (self.ps.density[p_i] ** 2)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                ret_i += self.ps.m[p_j] * (x[p_i] * inv_density_2 + x[p_j] / (self.ps.density[p_j] ** 2)) * self.ps.fluid_neighbors_values[p_i, j]

            ret[p_i] = ret_i


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
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            grad[p_i] = (self.dfdt[p_i] / self.Aii[p_i]) * grad[p_i]


    @ti.kernel
    def compute_Ax(self, Ax: ti.template(), Jx: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):

            Jx[p_i] = 0.0
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            if self.dfdt[p_i] >= 0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                    if self.ps.material[p_j] == self.ps.material_fluid:
                        Jx[p_i] += (x[p_i] - x[p_j]).dot(val_ij)
                    else:
                        Jx[p_i] += x[p_i].dot(val_ij)

            Jx[p_i] *= (self.dfdt[p_i] * self.k[p_i])

        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            Ax[p_i] = self.ps.m[p_i] * x[p_i]

        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            if self.dfdt[p_i] >= 0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                    Ax[p_i] += Jx[p_i] * val_ij

                    if self.ps.material[p_j] == self.ps.material_fluid:
                        Ax[p_j] -= Jx[p_i] * val_ij

    @ti.kernel
    def compute_Ax2(self, Ax: ti.template(), Jx: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            Ax[p_i] = 0.0

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    Jx[p_i] += (x[p_i] - x[p_j]).dot(val_ij)
                else:
                    Jx[p_i] += x[p_i].dot(val_ij)

            Jx[p_i] *= self.dfdt[p_i] / self.Aii[p_i]


        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                Ax[p_i] += Jx[p_i] * val_ij

                if self.ps.material[p_j] == self.ps.material_fluid:
                    Ax[p_j] -= Jx[p_i] * val_ij

    @ti.kernel
    def apply_precondition(self, dx: ti.template(), Hii: ti.template(), grad: ti.template()):

        for p_i in ti.grouped(self.ps.x):

            dx[p_i] = ti.math.vec3(0.0)
            I3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            H = self.ps.m[p_i] * I3x3 + Hii[p_i]
            dx[p_i] = H.inverse() @ grad[p_i]

    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> float:

        ret = 0.0
        for i in a:
            if self.ps.material[i] != self.ps.material_fluid:
                continue

            ret += ti.math.dot(a[i], b[i])

        return ret

    @ti.kernel
    def add(self, ret: ti.template(), v0: ti.template(), scale: float, v1: ti.template()):
        for i in ret:

            if self.ps.material[i] != self.ps.material_fluid:
                continue

            ret[i] = v0[i] + scale * v1[i]


    @ti.kernel
    def update_coeff(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            self.k[p_i] = self.dfdt[p_i] * ((self.c[p_i] / self.Aii[p_i]) + self.p[p_i])


    def PCG(self):

        x = self.a
        x.fill(0.0)
        z = self.z_pcg
        r = self.r_pcg
        p = self.p_pcg
        Ap = self.Ap
        b = self.tmp
        Jp = self.dp

        r.copy_from(b)
        self.apply_precondition(z, self.Hii, r)

        rz_old = dot(r, z)
        pcg_iter = 0
        if rz_old > 1e-12:
            p.copy_from(z)
            for _ in range(self.max_iteration_pcg):
                self.compute_Ax(Ap, Jp, p)
                pAp = dot(p, Ap)
                if pAp < 0.0:
                    print("Warning: non-positive definite matrix!")

                alpha = rz_old / pAp

                self.add(x, x, alpha, p)
                self.add(r, r, -alpha, Ap)

                err = self.dot(r, r)

                if self.print_pcg_error:
                    print(f"PCG error: {err}")

                self.apply_precondition(z, self.Hii, r)
                if err <  pow(10, -self.tol_pcg) or pcg_iter >= self.max_iteration_pcg:
                    break

                pcg_iter += 1

                rz_new = self.dot(r, z)
                beta = rz_new / rz_old
                self.add(p, z, beta, p)
                rz_old = rz_new


        if self.print_pcg_iter:
            print(f"PCG iter: {pcg_iter}")

    @ti.kernel
    def compute_f(self, ret: ti.template(), x: ti.template(), eps: float):

        for p_i in ti.grouped(x):

            ret[p_i] = 0.0
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            if x[p_i] >= 0.0:
                a_i = eps * self.ps.density0[p_i]
                ret[p_i] = ti.sqrt(ti.pow(x[p_i], 2) + a_i ** 2) - a_i



    @ti.kernel
    def compute_f_derivative(self, ret: ti.template(), x: ti.template(), eps: float):

        # a = 0
        for p_i in ti.grouped(self.ps.x):
            ret[p_i] = 0.0

            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            if x[p_i] >= 0.0:
                a_i = eps * self.ps.density0[p_i]
                ret[p_i] = x[p_i] / ti.sqrt(ti.pow(x[p_i], 2) + a_i ** 2)

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
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            self.c[p_i] = (self.ps.density[p_i] - self.ps.density0[p_i])

    @ti.kernel
    def compute_d(self):

        for p_i in ti.grouped(self.ps.x):

            self.d_v[p_i] = ti.math.vec3(0.0)
            self.d_l[p_i] = 0.0

            if self.ps.material[p_i] != self.ps.material_fluid:
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

            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            ret_v[p_i] = x_v[p_i]
            self.tmp[p_i] = x_v[p_i] / self.ps.m[p_i]

        for p_i in ti.grouped(self.ps.x):

            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            ret_l[p_i] = x_l[p_i]

            if self.dfdt[p_i] > 0.0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    # val = ti.cast(self.ps.material[p_i], float)
                    if self.ps.material[p_i] == self.ps.material_fluid:
                        if self.ps.material[p_j] == self.ps.material_fluid:
                            ret_l[p_i] -=self.ps.m[p_j] * (self.tmp[p_i] - self.tmp[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                        else:
                            ret_l[p_i] -= (self.ps.m[p_j] * self.tmp[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

        # step 2
        for p_i in ti.grouped(self.ps.x):

            # ret_v[p_i] = ti.math.vec3(0.0)
            # ret_l[p_i] = 0.0

            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            ret_v[p_i] =     ret_v[p_i] / self.ps.m[p_i]
            ret_l[p_i] = - 0.5 * (self.dfdt[p_i] / self.Aii[p_i]) * ret_l[p_i]

        # step 3
        for p_i in ti.grouped(self.ps.x):

            self.tmp[p_i] = ti.math.vec3(0.0)
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            # num_f += 1

            if self.dfdt[p_i] > 0.0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    # val = ti.cast(self.ps.material[p_i], float)
                    if self.ps.material[p_i] == self.ps.material_fluid:
                        if self.ps.material[p_j] == self.ps.material_fluid:
                            self.tmp[p_i] += (self.ps.m[p_j] * ret_l[p_i] + self.ps.m[p_i] * ret_l[p_j]) * self.ps.fluid_neighbors_values[p_i, j]
                        else:
                            self.tmp[p_i] += (self.ps.m[p_j] * ret_l[p_i]) * self.ps.fluid_neighbors_values[p_i, j]


            ret_v[p_i] = ret_v[p_i] - self.tmp[p_i] / self.ps.m[p_i]

    @ti.kernel
    def compute_B_x(self, ret_v: ti.template(), ret_l: ti.template(), x_v: ti.template(), x_l: ti.template()):

        #B    v   = Mv + J^tr * lambda
        #B lambda = Jv
        for p_i in ti.grouped(self.ps.x):

            ret_v[p_i] = ti.math.vec3(0.0)
            ret_l[p_i] = 0.0

            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            ret_v[p_i] = self.ps.m[p_i] * x_v[p_i]

            if self.dfdt[p_i] > 0.0:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    if self.ps.material[p_i] == self.ps.material_fluid:
                        if self.ps.material[p_j] == self.ps.material_fluid:
                            ret_v[p_i] += (self.ps.m[p_j] * x_l[p_i] + self.ps.m[p_i] * x_l[p_j]) * self.ps.fluid_neighbors_values[p_i, j]
                        else:
                            ret_v[p_i] += (self.ps.m[p_j] * x_l[p_i]) * self.ps.fluid_neighbors_values[p_i, j]

                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    if self.ps.material[p_j] == self.ps.material_fluid:
                        ret_l[p_i] += self.ps.m[p_j] * (x_v[p_i] - x_v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                    else:
                        ret_l[p_i] += self.ps.m[p_j] * (x_v[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

    @ti.kernel
    def evaluateConstraints(self, x_v: ti.template(), x_l: ti.template()) -> int:

        flag = 0
        for p_i in ti.grouped(self.ps.x):

            self.Jx[p_i] = 0.0
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_fluid:
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

    def constant_density_solve(self):

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
        c = self.c
        g = self.tmp
        P = self.Hii
        p = self.a

        opt_iter = 0
        for _ in range(self.max_iteration_opt):

            self.compute_J_x(Jd, d)
            add(self.t, Jd, 1.0, c)
            self.compute_f(self.f, self.t, self.eps)
            self.compute_f_derivative(self.dfdt, self.t, self.eps)
            self.compute_JtJ()
            coef_wise_mul(self.f, self.f, self.k)
            self.compute_J_tr_x(g, self.f)

            #Ours
            if self.smooth_max:
                if self.use_pcg:
                    self.PCG()
                else:
                    self.apply_precondition(p, P, g)

                self.add(d, d, -1.0, p)

            #Bender et al. 2014 (Constant density solver OF DFSPH)
            else:
                coef_wise_div(p, g, self.ps.m)
                self.add(d, d, -self.omega, p)


            err = inf_norm(p)
            if self.print_opt_error:
                print(f"opt iter: {err}")

            if (err < pow(10, -self.tol_opt)) and opt_iter > 1:
                break

            opt_iter += 1

        if self.print_elapsed_time:
            print("elapsed time: TODO [ms]")

        if self.print_opt_iter:
            print(f"opt iter: {opt_iter}")


        #x_n+1
        add(self.ps.x, self.ps.x, 1.0, self.dx)
        self.enforce_boundary_3D(self.ps.material_fluid)

        #v_n+1_tmp
        self.update_velocities(self.dt)


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

    def substep(self):

        self.ps.initialize_particle_system()
        self.ps.search_neighbours(self.ps.x)
        # self.ps.initialize_boundary_neighbors()
        self.ps.initialize_boundary_particles()

        dt_original = self.dt

        if self.cfl:
            v_max = inf_norm(self.ps.v)
            dt_upper_bound = 0.4 * (self.ps.particle_diameter / (v_max + 1e-12))
            dt_lower_bound = 0.001
            self.dt = ti.max(dt_lower_bound, ti.min(dt_upper_bound, self.dt))
            print(f"use smaller time step: {self.dt}")

        self.compute_non_pressure_forces()
        self.advect_velocity(self.dt)

        self.constant_density_solve()
        #
        # if self.enable_DF:
        #     self.divergence_free_solve()

        # error = self.measure_divergence(self.ps.v)
        # self.measure_divergence(self.ps.v, self.dt)
        self.dt = dt_original  # Reset dt to original value after substep