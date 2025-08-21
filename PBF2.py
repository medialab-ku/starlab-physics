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
        self.lda = self.ps.pressure
        self.method = 1
        self.iisph_vanilla = False 
        self.num_substep = self.ps.cfg.get_cfg("numSubstepping")

        self.adaptive_step_size = False
        self.gauss_newton_pcg = True
        self.print_info = True 
        self.tol = 2
        self.omega = 0.5 
        self.cfl = False
        self.max_iteration = 1000
        self.tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.grad = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.dx = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.dx_adv = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.dp   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.c   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)

        self.Aii = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.Dii = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.Hii = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.Ap  = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.x   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.p   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.y   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.b   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.z_pcg   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.r_pcg   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)


        self.Ap_b   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        # self.x   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.p_pcg    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.y_b    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.b_b    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.z_pcg    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.r_pcg    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.grad_b = ti.field(dtype=float, shape=self.ps.fluid_particle_num)

        self.stats_iter = 0
        self.stats_pcg_iter = 0
        print("method: PBF2")
        print("Available methods: 0=ProjectedJacobi, 1=ADMM, 2=Barrier, 3=NormalEquations")

        self.matrix_type = 0
        
        # Iteration logging system
        self.enable_logging = False
        self.iteration_log = []  # Store [frame, matrix_type, iterations]
        self.current_frame = 0
     


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

        # print(self.ps.m[0] * self.cubic_kernel(0.0))
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
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
            # self.ps.density[p_i] *= self.density_0

    @ti.kernel
    def compute_pressure_pbf(self) -> float:
        
        ret = 0.0
        eps = 1e-3 
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            Jdx_i = 0.0
            Aii = 0.0
            Hii = ti.math.mat3(0.0)
            J_ii = ti.math.vec3(0.0)
            dx_i = self.ps.x[p_i] - self.ps.y[p_i]
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # Fluid neighbors
                dx_j = self.ps.x[p_j] - self.ps.y[p_i]
                grad_ij = self.ps.fluid_neighbors_values[p_i, j]
                J_ij = self.ps.m[p_j] * grad_ij


                Hii += J_ij.outer_product(J_ij)
                if self.ps.material[p_j] == self.ps.material_fluid:
                    Aii += J_ij.dot(J_ij) / self.ps.m[p_j]

                J_ii -= J_ij

                if self.ps.material[p_j] == self.ps.material_fluid:
                    Jdx_i += self.ps.m[p_j] * (dx_i - dx_j).dot(grad_ij)
                else:
                    Jdx_i += self.ps.m[p_j] * (dx_i).dot(grad_ij)

            Aii += J_ii.dot(J_ii) / self.ps.m[p_i]
            Hii += J_ii.outer_product(J_ii)
            c = self.ps.density[p_i] - self.ps.density0[p_i]
            ret += (ti.max(c, 0.0) / self.ps.density0[p_i]) 
            self.c[p_i] = c
            self.Hii[p_i] = Hii 
            self.Aii[p_i] = Aii + eps
            self.p[p_i] = ti.max(c, 0.0) / self.Aii[p_i]

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
            f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
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
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            Aii = 0.0
            J_ii = ti.math.vec3(0.0)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                J_ij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]

                if self.ps.material[p_j] == self.ps.material_fluid:
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
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    ret_i += self.ps.m[p_j] * (x[p_i] - x[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])
                else:
                    ret_i += self.ps.m[p_j] * (x[p_i]).dot(self.ps.fluid_neighbors_values[p_i, j])

            ret[p_i] = ret_i

    @ti.kernel
    def compute_J_tr_x(self, ret: ti.template(), x: ti.template()):

        # num_f = 0
        for p_i in ti.grouped(x):
            ret[p_i] = ti.math.vec3(0.0)
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            # num_f += 1
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # val = ti.cast(self.ps.material[p_i], float)
                if self.ps.material[p_i] == self.ps.material_fluid:
                    if self.ps.material[p_j] == self.ps.material_fluid:
                        ret[p_i] += (self.ps.m[p_j] * x[p_i] + self.ps.m[p_i] * x[p_j]) * self.ps.fluid_neighbors_values[p_i, j]
                    else:
                        ret[p_i] += (self.ps.m[p_j] * x[p_i]) * self.ps.fluid_neighbors_values[p_i, j]

        # print(num_f)
            # ret[p_i] = ret_i

    @ti.kernel
    def compute_b(self, b: ti.template(), v: ti.template(), dt: float):

        dtSq = dt ** 2
        for p_i in ti.grouped(b):
            div_i = 0.0
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                div_i += self.ps.m[p_j] * (v[p_i] - v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])

            b[p_i] = (self.ps.density[p_i] + dt * div_i - self.ps.density0[p_i]) / dtSq

    @ti.kernel
    def measure_error(self, v: ti.template(), dt: float) -> float:

        avg_error = 0.0
        for p_i in ti.grouped(v):
            div_i = 0.0
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                div_i += self.ps.m[p_j] * (v[p_i] - v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])

            avg_error = ti.max(self.ps.density[p_i] + dt * div_i - self.ps.density0[p_i], 0.0) / self.ps.density0[p_i]

        return avg_error


    def pressure_solve(self):
        
        if self.method == 0:
            self.IISPH()



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

    def IISPH(self):
        
        self.compute_density()
        # self.precompute_values()
        self.ps.v.copy_from(self.ps.v_adv)
        self.compute_Aii(self.iisph_vanilla)
        self.compute_b(self.b, self.ps.v, self.dt)

        tol = pow(10, -self.tol)
        self.p.fill(0.0)
        iter = 0
        for _ in range(self.max_iteration):
        
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

            self.compute_J_x(self.Ap, self.tmp)
            add(self.r_pcg, self.b, -1.0, self.Ap)

            coef_wise_op(self.dp, self.r_pcg, self.Aii, 1)

    
            add(self.p, self.p, self.omega, self.dp)  # Initialize p with b
            max(self.p)  # Ensure non-negativity
        
        
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
    def apply_precondition(self, dx: ti.template(), Hii: ti.template(), grad: ti.template()):

        I3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            
            H = self.ps.m[p_i] * I3x3 + Hii[p_i] / self.Aii[p_i]
            dx[p_i] = H.inverse() @ grad[p_i]


    def PBF(self):

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

        for _ in range(self.max_iteration):

            self.compute_density()

            # original problem: (JM^-1Jt) * p = c
            # PBD approximation: p = diag(JM^-1Jt)^-1 * max(c, 0)
            error = self.compute_pressure_pbf()
            if error < tol and iter > 1 or iter == self.max_iteration:
                if self.print_info:
                    print(f" converged iter: {iter}. error: {error}")
                break
            
            # dx = self.tmp
            # self.compute_J_tr_x(self.tmp, self.p)
            # coef_wise_op(dx, dx, self.ps.m, 1)
            
            step_size = 0.5
            if self.gauss_newton_pcg:

                add(self.dx_adv, self.ps.y, -1.0, self.ps.x)
                self.compute_J_x(self.dp, self.dx_adv)
                max(self.dp)
                max(self.c)
                # add(self.c, self.c, 1.0, self.dp)  # Initialize p with b
                coef_wise_op(self.p, self.c, self.Aii, 1)
                self.compute_J_tr_x(self.tmp, self.p)

                self.compute_gradient(self.grad, False)
                self.apply_precondition(self.dx, self.Hii, self.grad)
                step_size = 1.0
            else:
                self.compute_J_tr_x(self.tmp, self.p)
                coef_wise_op(self.dx, self.tmp, self.ps.m, 1)

            if self.adaptive_step_size:
                  div = self.dp 
                  self.compute_J_x(div, self.dx)
                  aTa = dot2(div, div)
                  aTb = dot2(div, self.c)

                  k = 1e8
                  alpha = (k * aTb) / (k * aTa + 1.0)
                  step_size = ti.min(1.0, alpha)

            # x^k+1 = x^k - step_size(=0.5) * M^-1 J^t * p
            add(self.ps.x, self.ps.x, -step_size, self.dx)

            iter += 1

        self.update_velocities(self.dt)

        

    def substep(self):
        
        self.ps.search_neighbours(self.ps.x)


        dt_original = self.dt

        if self.cfl:
            v_max = inf_norm(self.ps.v)
            dt_upper_bound = 0.4 * (self.ps.particle_diameter / (v_max + 1e-12))
            dt_lower_bound = 0.001
            self.dt = ti.max(dt_lower_bound, ti.min(dt_upper_bound, self.dt))
            print(f"use smaller time step: {self.dt}")

        self.compute_non_pressure_forces()
        self.advect_velocity(self.dt)    
        if self.method == 0:
            self.IISPH()
        elif self.method == 1:
            self.PBF()

        self.dt = dt_original  # Reset dt to original value after substep
