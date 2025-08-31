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

        self.nablaWij = self.cubic_kernel_derivative
        self.Wij = self.cubic_kernel
        self.lda = self.ps.pressure
        self.method = 1
        self.iisph_vanilla = False 
        self.num_substep = self.ps.cfg.get_cfg("numSubstepping")

        self.adaptive_step_size = False
        self.gauss_newton_pcg = True
        self.divergence_free_solve = False
        self.print_info = True
        self.tol = 2
        self.omega = 0.5 
        self.cfl = False
        self.max_iteration_opt = 1000
        self.max_iteration_pcg = 1000
        self.pcg_tol = 1e-4
        self.tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.grad = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.dx = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.dx_adv = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.x_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.dp   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.c   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.p   = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Aii = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.Dii = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.x   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.y   = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.b   = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Jx = ti.field(dtype=float, shape=self.ps.particle_max_num)
        self.r_jacobi = ti.field(dtype=float, shape=self.ps.particle_max_num)

        self.Hii     = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.particle_max_num)
        self.Ap      = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.b_pcg   = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.z_pcg   = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.r_pcg   = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.p_pcg   = ti.Vector.field(n=3, dtype=float, shape=self.ps.particle_max_num)
        self.stats_iter = 0
        self.stats_pcg_iter = 0
        # print("method: PBF2")
        # print("Available methods: 0=ProjectedJacobi, 1=ADMM, 2=Barrier, 3=NormalEquations")
        self.matrix_type = 0
        
        # Iteration logging system
        self.enable_logging = False
        self.iteration_log = []  # Store [frame, matrix_type, iterations]
        self.current_frame = 0


    # @ti.kernel
    # def initialize_boundary_particles(self):
    #     for p_i in ti.grouped(self.x):
    #         sum_Wij = 0.0
    #         for j in range(self.ps.fluid_neighbors_num[p_i]):
    #             p_j = self.ps.fluid_neighbors[p_i, j]
    #             if self.ps.material[p_i] == self.ps.material_solid or self.ps.material[p_j] == self.ps.material_solid:
    #                 continue
    #             sum_Wij += self.Wij((self.ps.x[p_i] - self.ps.x[p_j]).norm())

    #         self.ps.m[p_i] = self.ps.density0[p_i] / sum_Wij

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
        # for p_i in range(self.ps.particle_num[None]):

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

            # if self.ps.is_dynamic_rigid_body(p_i):
            #     # self.ps.density[p_i] *= self.density_0
            #     den = 0.0
            #     self.ps.density[p_i] = self.ps.density0[p_i]
            #     x_i = self.ps.x[p_i]
            #     static_cnt = 0
            #     # self.ps.density[p_i] = self.ps.m[p_i] * self.cubic_kernel(0.0)
            #     for j in range(self.ps.fluid_neighbors_num[p_i]):
            #         p_j = self.ps.fluid_neighbors[p_i, j]
            #         if self.ps.material[p_j] != self.ps.material_solid:
            #             continue

            #         if self.ps.is_static_rigid_body(p_j) and (self.ps.object_id[p_j] != self.ps.object_id[p_i]):
            #             den += self.ps.m[p_j] * self.cubic_kernel((x_i - self.ps.x[p_j]).norm())
            #             static_cnt += 1

            #     self.ps.density[p_i] += den
            #     if self.ps.is_dynamic_rigid_body(p_i):
            #         num_dynamic_rigid += 1
            #         total_static_neighbors += static_cnt
            #         if self.ps.density[p_i] > self.ps.density0[p_i]:
            #             cnt += 1


        avg_static_neighbors = 0.0
        if num_dynamic_rigid > 0:
            avg_static_neighbors = total_static_neighbors / num_dynamic_rigid
        # print(f"num high density rigid body: {cnt}, avg static neighbors per dynamic rigid: {avg_static_neighbors}")

    @ti.kernel
    def compute_pressure_pbf(self) -> float:
        
        ret = 0.0
        eps = 1e-3
        I3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            self.Aii[p_i] = eps
            J_ii = ti.math.vec3(0.0)
            self.c[p_i] = 0.0
            self.p[p_i] = 0.0

            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # Fluid neighbors
                grad_ij = self.ps.fluid_neighbors_values[p_i, j]
                J_ij = self.ps.m[p_j] * grad_ij

                if self.ps.material[p_j] == self.ps.material_fluid:
                    self.Aii[p_i] += J_ij.dot(J_ij) / self.ps.m[p_j]
                
                J_ii -= J_ij
            
            self.Aii[p_i] += J_ii.dot(J_ii) / self.ps.m[p_i]
            c = self.ps.density[p_i] - self.ps.density0[p_i]
            ret += (ti.max(c, 0.0) / self.ps.density0[p_i])
            self.c[p_i] = c
            self.p[p_i] = ti.max(c, 0.0) / self.Aii[p_i]

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            self.Hii[p_i] = self.ps.m[p_i] * I3x3
            J_ii = ti.math.vec3(0.0)

            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                grad_ij = self.ps.fluid_neighbors_values[p_i, j]
                J_ij = self.ps.m[p_j] * grad_ij
                if self.ps.material[p_j] == self.ps.material_fluid:
                    self.Hii[p_i] += J_ij.outer_product(J_ij) / self.Aii[p_i]

                J_ii -= J_ij
            self.Hii[p_i] += J_ii.outer_product(J_ii) / self.Aii[p_i]

        ret /= self.ps.fluid_particle_num
        return ret
    

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
                # self.ps.x[p_i] += self.dt * self.ps.v[p_i]

    @ti.kernel
    def advect_position(self, dt: float):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                # self.ps.v[p_i] += self.dt * self.ps.acceleration[p_i]
                self.ps.x[p_i] += dt * self.ps.v[p_i]
            


    @ti.kernel
    def compute_Aii(self, iisph_vanilla: bool):

        eps = 1e-3

        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                continue

            Aii = 0.0
            J_ii = ti.math.vec3(0.0)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                J_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]

                if self.ps.is_dynamic[p_j]:
                    Aii += J_ij.dot(J_ij) / self.ps.m[p_j]

                J_ii -= J_ij
            Aii += J_ii.dot(J_ii) / self.ps.m[p_i]

            if iisph_vanilla:
                 Dii = (self.ps.m[p_i] / self.ps.density[p_i] ** 2)
                 self.Dii[p_i] = Dii
                 self.Aii[p_i] = Dii * Aii + eps
            else:
                self.Aii[p_i] = Aii + eps
 

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

            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    ret[p_i] += self.ps.m[p_j] * (x[p_i] - x[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    ret[p_i] += self.ps.m[p_j] * (x[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            # ret[p_i] = ret_i

    @ti.kernel
    def num_negative_p(self, p: ti.template()) -> int:
        cnt = 0
        for i in p:
            if p[i] < 0.0:
                cnt += 1
        return cnt  
    

    @ti.kernel
    def compute_J_tr_x(self, ret: ti.template(), x: ti.template()):

        # num_f = 0
        for p_i in ti.grouped(x):
            ret[p_i] = ti.math.vec3(0.0)
            if not self.ps.is_dynamic[p_i]:
                continue

            # num_f += 1
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # val = ti.cast(self.ps.material[p_i], float)
                if self.ps.is_dynamic[p_j]:
                    ret[p_i] += (self.ps.m[p_j] * x[p_i] + self.ps.m[p_i] * x[p_j]) * self.ps.fluid_neighbors_values[p_i, j]
                else:
                    ret[p_i] += (self.ps.m[p_j] * x[p_i]) * self.ps.fluid_neighbors_values[p_i, j]

    
    @ti.kernel
    def compute_J_tr_x_active(self, ret: ti.template(), x: ti.template()):

        # num_f = 0
        for p_i in ti.grouped(x):
            ret[p_i] = ti.math.vec3(0.0)
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    gradWij = self.ps.fluid_neighbors_values[p_i, j]
                    ret[p_i] += x[p_i] * self.ps.m[p_j] * gradWij
                    ret[p_j] -= x[p_i] * self.ps.m[p_j] * gradWij
                else:
                    ret[p_i] += x[p_i] * self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]


        # print(num_f)
            # ret[p_i] = ret_i

    @ti.kernel
    def compute_b(self, b: ti.template(), v: ti.template(), dt: float):

        dtSq = dt ** 2
        for p_i in ti.grouped(b):
            div_i = 0.0
            if not self.ps.is_dynamic[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
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
    def measure_divergence(self, v: ti.template(), dt: float) -> float:
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

        return dt*avg_error


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

            inv_density_2 = 1.0 / (self.ps.density[p_i] ** 2) 
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                ret_i += self.ps.m[p_j] * (x[p_i] * inv_density_2 + x[p_j] / (self.ps.density[p_j] ** 2)) * self.ps.fluid_neighbors_values[p_i, j]

            ret[p_i] = ret_i


    def divergence_free_sovle_IISPH(self):

        self.compute_density()
        # self.precompute_values()
        self.ps.v.copy_from(self.ps.v_adv)
        self.v_tmp.copy_from(self.ps.v)
        self.compute_Aii(self.iisph_vanilla)
        # self.compute_b(self.b, self.ps.v, self.dt)
        self.compute_b_div(self.b, self.ps.v_adv, self.dt)

        tol = pow(10, -self.tol+1)
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
        
        self.compute_density()
        # self.precompute_values()
        self.ps.v.copy_from(self.ps.v_adv)
        self.compute_Aii(self.iisph_vanilla)
        self.compute_b(self.b, self.ps.v, self.dt)

        tol = pow(10, -self.tol)
        self.p.fill(0.0)
        iter = 0
        for _ in range(self.max_iteration_opt):
        
            if self.iisph_vanilla:
                coef_wise_mul(self.y, self.Dii, self.p)
                self.compute_J_tr_x(self.tmp, self.y)
            else:
                self.compute_J_tr_x(self.tmp, self.p)
                
                
            coef_wise_op(self.tmp, self.tmp, self.ps.m, 1)
            add(self.ps.v, self.ps.v_adv, -self.dt, self.tmp)
            error = self.measure_error(self.ps.v, self.dt)

            if error < tol and iter > 2:

                print(f" converged iter: {iter}. error: {error}")
                break 
            
            iter += 1 

            self.compute_J_x(self.Jx, self.tmp)
            add(self.r_jacobi, self.b, -1.0, self.Jx)

            coef_wise_op(self.dp, self.r_jacobi, self.Aii, 1)

    
            add(self.p, self.p, self.omega, self.dp)  # Initialize p with b
            max(self.p)  # Ensure non-negativity
        
        # self.apply_rigid_pressure(self.dt)
        self.advect_position(self.dt)




    @ti.kernel
    def compute_gradient(self, grad: ti.template(), adv: bool):


        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            grad[p_i] = self.tmp[p_i]
            if adv:
                grad[p_i] += self.ps.m[p_i] * (self.ps.x[p_i] - self.ps.y[p_i])


    @ti.kernel
    def compute_Ax(self, Ax: ti.template(), Jx: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            Jx[p_i] = 0.0

            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    Jx[p_i] += (x[p_i] - x[p_j]).dot(val_ij)
                else:
                    Jx[p_i] += x[p_i].dot(val_ij)

            Jx[p_i] /= self.Aii[p_i]

        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            Ax[p_i] = self.ps.m[p_i] * x[p_i]

        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                val_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                Ax[p_i] += Jx[p_i] * val_ij

                if self.ps.material[p_j] == self.ps.material_fluid:
                    Ax[p_j] -= Jx[p_i] * val_ij


    @ti.kernel
    def apply_precondition(self, dx: ti.template(), Hii: ti.template(), grad: ti.template()):

        # I3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            H = Hii[p_i]
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
            

    def constant_density_solve_PBF(self):

        # objective: min_x ||x - y||^2_{M/h^2} s.t. c(x) <= 0
        # linearization:  min_x ||x^k+1 - y||^2_{M/h^2} s.t. c(x^k) + J * (x^k+1 - x^k) <= 0
        # (1) M/h^2 * (x^k + dx - y) + J^k^T p = 0
        # (2)  J^k * dx = 0

        tol = pow(10, -self.tol)
        # tol = 0.1
        self.ps.x_old.copy_from(self.ps.x)

        # y = x_n + dt * v_n + dt * M^-1 * f_ext
        add(self.ps.y, self.ps.x, self.dt, self.ps.v_adv)

        # x^0 = y (warm start)
        # A = dtSq * JM^-1J^t
        # p = c / diag(A)

        # IISPH: x^k+1 = y   - M^-1 J^t * p (with full linear solve for p, Ap = b, single iteration )
        # PBF  : x^k+1 = x^k - M^-1 J^t * p (with approximation, diag(A) * p = b, multiple iterations)
        # ???  : (M + H) * dx = M * (y - x) - J^t p, x^k+1 = x^k + alpha * dx, what is best H?

        self.ps.x.copy_from(self.ps.y)
        iter = 0

        for _ in range(self.max_iteration_opt):

            self.compute_density()

            # original problem: (JM^-1Jt) * p = c
            # PBD approximation: p = diag(JM^-1Jt)^-1 * max(c, 0)
            error = self.compute_pressure_pbf()
            if error < tol and iter > 1 or iter == self.max_iteration_opt:
                if self.print_info:
                    print(f" converged iter: {iter}. error: {error}")
                break

            # dx = self.tmp
            # self.compute_J_tr_x(self.tmp, self.p)
            # coef_wise_op(dx, dx, self.ps.m, 1)

            step_size = 0.5
            if self.gauss_newton_pcg:
                
                # Preconditioned Conjugate Gradient (PCG) Pseudo Code:
                # Given: A*x = b, preconditioner M
                # 1. r_0 = b - A*x_0  (initial residual)
                # 2. z_0 = M^(-1) * r_0  (apply preconditioner)
                max(self.c)
                dx = self.dx
                self.dx.fill(0.0)
                z = self.z_pcg
                r = self.r_pcg
                p = self.p_pcg
                Ap = self.Ap
                b = self.b_pcg
                Jp = self.dp  
                
                coef_wise_op(self.p, self.c, self.Aii, 1)
                self.compute_J_tr_x_active(self.tmp, self.p)

                self.grad.fill(0.0)
                self.compute_gradient(self.grad, False)

                b.copy_from(self.grad)
                r.copy_from(b)
                self.apply_precondition(z, self.Hii, r)

                rz_old = self.dot(r, z)

                if rz_old > self.pcg_tol:
                #     # print(rz_old)
                # # 3. p_0 = z_0  (initial search direction)
                    p.copy_from(z)
                    pcg_iter = 1
                    for _ in range(self.max_iteration_pcg):

                        self.compute_Ax(Ap, Jp, p)
                        pAp = self.dot(p, Ap)
                        if pAp < 0.0:
                            print("Warning: non-positive definite matrix!")
                        # #     break
                        alpha = rz_old / pAp
                        self.add(dx, dx, alpha, p)
                        self.add(r, r, -alpha, Ap)
                        err = self.dot(r, r)
                        self.apply_precondition(z, self.Hii, r)

                        rz_new = dot(r, z)
                        if err < self.pcg_tol or pcg_iter >= self.max_iteration_pcg:
                            print(f"PCG iter: , {pcg_iter}: error: {err}")
                            break
                        pcg_iter += 1
                        beta = rz_new / rz_old
                        self.add(p, z, beta, p)
                        rz_old = rz_new

                step_size = 1.0


            else:
                self.compute_J_tr_x_active(self.tmp, self.p)
                coef_wise_op(self.dx, self.tmp, self.ps.m, 1)

            if self.adaptive_step_size:
                  div = self.dp
                  self.compute_J_x_active(div, self.dx)
                  aTa = dot2(div, div)
                  aTb = dot2(div, self.c)

                  k = 1e8
                  alpha = (k * aTb) / (k * aTa + 1.0)
                  step_size = ti.min(1.0, alpha)

            # x^k+1 = x^k - step_size(=0.5) * M^-1 J^t * p
            self.add(self.ps.x, self.ps.x, -step_size, self.dx)

            iter += 1

        self.update_velocities(self.dt)



    def substep(self):
        self.ps.initialize_particle_system()
        self.ps.search_neighbours(self.ps.x)
        # self.ps.initialize_boundary_neighbors()
        dt_original = self.dt

        if self.cfl:
            v_max = inf_norm(self.ps.v)
            dt_upper_bound = 0.4 * (self.ps.particle_diameter / (v_max + 1e-12))
            dt_lower_bound = 0.001
            self.dt = ti.max(dt_lower_bound, ti.min(dt_upper_bound, self.dt))
            print(f"use smaller time step: {self.dt}")

        self.compute_non_pressure_forces()
        self.advect_velocity(self.dt)

        #divergence-free condition solve
        if self.divergence_free_solve:
            if self.method == 0:
                self.divergence_free_sovle_IISPH()

        #constant density condition solve
        if self.method == 0:
            self.constant_density_solve_IISPH()
        elif self.method == 1:
            self.constant_density_solve_PBF()

        # self.measure_divergence(self.ps.v, self.dt)
        self.dt = dt_original  # Reset dt to original value after substep