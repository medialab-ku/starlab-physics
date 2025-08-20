import taichi as ti
import math
import os
import json
from sph_base import SPHBase
from math_utils import *

"""
PBF2 Solver - Position Based Fluids with Multiple Solver Options

This implementation provides several approaches to solve the implicit incompressible SPH system:

Method 0 (ProjectedJacobi): 
    Solves (J M⁻¹ Jᵀ) y = b where y = Dp
    - J = ∇ρ (density gradient operator)
    - M⁻¹ = inverse mass matrix
    - D = diagonal density-weighting matrix
    - Advantages: Simple, preserves symmetry naturally

Method 1 (ADMM): 
    Alternating Direction Method of Multipliers
    - Uses augmented Lagrangian approach
    - Good for complex constraints

Method 2 (Barrier): 
    Barrier function method
    - Uses PCG with barrier functions

Method 3 (NormalEquations): 
    Solves (Dᵀ J M⁻¹ Jᵀ D) p = Dᵀ b
    - Creates symmetric system by multiplying both sides by Dᵀ
    - Advantages: Guaranteed symmetric, can use symmetric solvers
    - Disadvantages: Squared condition number, potentially less stable
    - Implementation: Uses matrix-free operators for efficiency

The Normal Equations approach (Method 3) is particularly useful when you need:
1. A guaranteed symmetric system
2. To use symmetric solvers (CG, Jacobi, etc.)
3. To avoid the complexity of other symmetrization methods

Mathematical formulation:
Original system: (J M⁻¹ Jᵀ D) p = b
Normal equations: (Dᵀ J M⁻¹ Jᵀ D) p = Dᵀ b
"""

class PBF2Solver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        # self.exponent = 7.0
        # self.exponent = self.ps.cfg.get_cfg("exponent")
        #
        # self.stiffness = 50000.0
        # self.stiffness = self.ps.cfg.get_cfg("stiffness")

        self.surface_tension = 0.001
        self.dt = self.ps.cfg.get_cfg("timeStepSize")

        self.nablaWij = self.cubic_kernel_derivative
        self.lda = self.ps.pressure
        self.method = 1
        self.iisph_vanilla = False 
        self.num_substep = self.ps.cfg.get_cfg("numSubstepping")
        self.tol = 2
        self.omega = 0.5 
        self.cfl = True
        self.max_iteration = 1000

        self.tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.dp   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.c   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)

        self.Aii = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.Dii = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.Hii = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
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

        # Additional fields for Normal Equations approach
        self.tmp2 = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.b_normal = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.tmp_vec = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)

        self.stats_iter = 0
        self.stats_pcg_iter = 0
        print("method: PBF2")
        print("Available methods: 0=ProjectedJacobi, 1=ADMM, 2=Barrier, 3=NormalEquations")

        self.matrix_type = 0
        
        # Iteration logging system
        self.enable_logging = False
        self.iteration_log = []  # Store [frame, matrix_type, iterations]
        self.current_frame = 0
        
        # Spectral radius analysis
        self.spectral_radius_analyzer = None
        self.enable_spectral_analysis = False
        
    def initialize_spectral_radius_analysis(self):
        """Initialize spectral radius analyzer"""
        try:
            from spectral_radius_analysis import SpectralRadiusAnalyzer
            self.spectral_radius_analyzer = SpectralRadiusAnalyzer(self)
            self.enable_spectral_analysis = True
            print("Spectral radius analysis initialized")
        except ImportError as e:
            print(f"Warning: Could not initialize spectral radius analysis: {e}")
            self.enable_spectral_analysis = False


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
            J_ii = ti.math.vec3(0.0)
            dx_i = self.ps.x[p_i] - self.ps.y[p_i]
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                # Fluid neighbors
                dx_j = self.ps.x[p_j] - self.ps.y[p_i]
                grad_ij = self.ps.fluid_neighbors_values[p_i, j]
                J_ij = self.ps.m[p_j] * grad_ij

                if self.ps.material[p_j] == self.ps.material_fluid:
                    Aii += J_ij.dot(J_ij) / self.ps.m[p_j]

                J_ii -= J_ij

                if self.ps.material[p_j] == self.ps.material_fluid:
                    Jdx_i += self.ps.m[p_j] * (dx_i - dx_j).dot(grad_ij)
                else:
                    Jdx_i += self.ps.m[p_j] * (dx_i).dot(grad_ij)

            Aii += J_ii.dot(J_ii) / self.ps.m[p_i]
                
            c = ti.max(self.ps.density[p_i] - self.ps.density0[p_i], 0.0)
            ret += (c / self.ps.density0[p_i]) 
            self.c[p_i] = c
            self.Aii[p_i] = Aii + eps
            self.p[p_i] = c / (Aii + eps)

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
                print(f" converged iter: {iter}. error: {error}")
                break


            self.compute_J_tr_x(self.tmp, self.p)
            coef_wise_op(self.tmp, self.tmp, self.ps.m, 1)

            # x^k+1 = x^k - step_size(=0.5) * M^-1 J^t * p


            add(self.ps.x, self.ps.x, -0.5, self.tmp)

            iter += 1

        self.update_velocities(self.dt)

        

    def substep(self):
        
        
        self.ps.search_neighbours(self.ps.x)
        self.compute_non_pressure_forces()

        self.advect_velocity(self.dt)
        if self.method == 0:
            self.IISPH()
        elif self.method == 1:
            self.PBF()

