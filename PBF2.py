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
        self.method = 0
        self.iisph_vanilla = False 

        self.tol = 2
        self.toggle = True
        self.max_iteration = 1000

        self.tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)
        self.dp   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)

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


    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            ret += self.ps.m[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())

    @ti.func
    def compute_divergence_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        v_i = self.ps.v[p_i]
        y_i = self.ps.y[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            v_j = self.ps.v[p_j]
            y_j = self.ps.y[p_j]
            ret += (self.ps.m[p_i] / self.ps.density0[p_i]) * self.ps.m[p_j] * self.spiky_kernel_derivative(x_i - x_j).dot(v_i - v_j)
        # elif self.ps.material[p_j] == self.ps.material_solid:
        #     # Boundary neighbors
        #     ## Akinci2012
        #     x_j = self.ps.x[p_j]
        #     ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())

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
                den += self.ps.m[p_j] * self.cubic_kernel((x_i - x_j).norm())

            # self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            # self.ps.density[p_i] *= self.density_0

    @ti.kernel
    def compute_schur(self):
        # for p_i in range(self.ps.particle_num[None]):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            x_i = self.ps.x[p_i]

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                den += self.ps.m[p_j] * self.cubic_kernel((x_i - x_j).norm())

            # self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            # self.ps.density[p_i] *= self.density_0

    @ti.kernel
    def compute_divergence(self):
        # for p_i in range(self.ps.particle_num[None]):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_divergence_task, den)
            # self.div[p_i] = ti.max(den, 0.0)


    @ti.func
    def compute_lambdas_v_task(self, p_i, p_j, ret: ti.template()):


        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            # m_j = self.density_0 * self.ps.m_V[p_j]
            nabla_cij = (self.ps.m[p_i] / self.ps.density0[p_i]) * self.ps.m[p_j] * self.nablaWij(x_i - x_j)
            ret[3] += nabla_cij.dot(nabla_cij) /self.ps.m[p_j]

            for i in range(3):
                ret[i] -= nabla_cij[i]

    @ti.func
    def compute_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        lambda_i = self.ps.pressure[p_i]
        m_i = (self.density_0 * self.ps.m_V[p_i])
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            m_j = self.density_0 * self.ps.m_V[p_j]
            density_j = self.ps.density[p_j]  # TODO: The density_0 of the neighbor may be different when the fluid density is different
            lambda_j = self.ps.pressure[p_j]
            # Compute the pressure force contribution, Symmetric Formula
            ret += self.ps.m[p_j] * (lambda_i / self.ps.density0[p_i] + lambda_j / self.ps.density0[p_j]) * self.nablaWij(x_i - x_j)
        # elif self.ps.material[p_j] == self.ps.material_solid:
        #     # Boundary neighbors
        #     dpj = self.ps.pressure[p_i] / self.density_0 ** 2
        #     ## Akinci2012
        #     x_j = self.ps.x[p_j]
        #     # Compute the pressure force contribution, Symmetric Formula
        #     f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
        #           * self.cubic_kernel_derivative(x_i - x_j)
        #     ret += f_p
        #     if self.ps.is_dynamic_rigid_body(p_j):
        #         self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]

    @ti.func
    def compute_divergence_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        lambda_i = self.ps.pressure[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            m_j = self.density_0 * self.ps.m_V[p_j]

            lambda_j = self.ps.pressure[p_j]
            # Compute the pressure force contribution, Symmetric Formula
            ret += m_j * (lambda_i / self.ps.density0[p_i] + lambda_j / self.ps.density0[p_j]) * self.spiky_kernel_derivative(x_i - x_j)


    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
            self.ps.pressure[p_i] = self.stiffness * (ti.pow(self.ps.density[p_i] / self.density_0, self.exponent) - 1.0)
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i):
                continue
            dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dv)
            self.ps.acceleration[p_i] += dv

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
                Aii += J_ij.dot(J_ij) / self.ps.m[p_j]

                J_ii -= J_ij
            Aii += J_ii.dot(J_ii) / self.ps.m[p_i]

            if iisph_vanilla:
                 Dii = (self.ps.m[p_i] / self.ps.density[p_i] ** 2)
                 self.Dii[p_i] = Dii
                 self.Aii[p_i] = Dii * Aii + eps
            else:
                self.Aii[p_i] = Aii + eps
 

    @ti.func
    def compute_lambdas_task2(self, p_i, p_j, ret: ti.template()):

        # schur = 0.0
        m_i = (self.density_0 * self.ps.m_V[p_i])
        # dc_dxi = ti.math.vec3(0.0)
        x_i = self.ps.x[p_i]
        # Fluid neighbors
        # dc_drho_i = self.density_0 * self.ps.m_V[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            # m_j = self.density_0 * self.ps.m_V[p_j]
            nabla_cij = (1.0 / self.ps.density0[p_i]) * self.ps.m[p_j] * self.nablaWij(x_i - x_j)
            # dc_dxi -= nabla_cij
            ret[3] += nabla_cij.dot(nabla_cij) / self.ps.m[p_j]

            for i in range(3):
                ret[i] -= nabla_cij[i]

    @ti.kernel
    def compute_lambdas_p2(self) -> float:

        eps = 1e-6
        avg_density_err = 0.0

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            c = (1.0 / self.ps.density0[p_i]) * ti.max(self.ps.density[p_i] - self.ps.density0[p_i], 0.0)
            avg_density_err += c
            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])
            # ret = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_lambdas_task2, ret)

            schur = ret[3]
            dc_dxi = ti.Vector([ret[0], ret[1], ret[2]])
            schur += dc_dxi.dot(dc_dxi) / self.ps.m[p_i]
            self.ps.pressure[p_i] = - c / (schur + eps)

        avg_density_err /= self.ps.fluid_particle_num
        return avg_density_err

    @ti.kernel
    def compute_lambdas_v(self) -> float:

        eps = 1e-3
        avg_density_err = 0.0

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            # m_i = (self.density_0 * self.ps.m_V[p_i])
            avg_density_err += self.div[p_i]
            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])
            # ret = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_lambdas_v_task, ret)

            schur = ret[3]
            dc_dxi = ti.Vector([ret[0], ret[1], ret[2]])
            schur += dc_dxi.dot(dc_dxi) / self.ps.m[p_i]
            self.ps.pressure[p_i] = -ti.max(self.div[p_i], 0.0) / (schur + eps)

        avg_density_err /= self.ps.fluid_particle_num
        return avg_density_err

    @ti.kernel
    def step_forward_x(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i):
                continue

            # m_i = (self.density_0 * self.ps.m_V[p_i])
            dx = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dx)
            self.ps.x[p_i] += dx

    @ti.kernel
    def step_forward_x2(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i):
                continue

            m_i = (self.density_0 * self.ps.m_V[p_i])
            dx = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dx)
            self.ps.x[p_i] += dx / m_i

    @ti.kernel
    def step_forward_v(self):

        # print("test")
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i):
                continue
        #
            m_i = (self.density_0 * self.ps.m_V[p_i])
            dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_divergence_forces_task, dv)
            self.ps.v[p_i] += dv



    @ti.kernel
    def update_velocities(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] = (self.ps.x[p_i] - self.ps.x_old[p_i])/ self.dt

    @ti.kernel
    def project(self, p: ti.template()):
        for p_i in ti.grouped(p):
            p[p_i] = ti.max(p[p_i], 0.0)

    @ti.kernel
    def update_pressure_acceleration(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                v_tmp = (self.ps.x[p_i] - self.ps.x_old[p_i]) / self.dt
                self.ps.acceleration[p_i] += (v_tmp - self.ps.v_old[p_i]) / self.dt


    @ti.kernel
    def apply_precondition(self, z: ti.template(), Aii: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            # z[p_i] = x[p_i] / Aii[p_i]
            denom = ti.max(Aii[p_i], 1e-12)
            z[p_i] = x[p_i] / denom

    @ti.kernel
    def compute_matrix_free_Ax_step_0(self, x: ti.template()):

        for p_i in ti.grouped(x):
            xi = x[p_i]
            x[p_i] = (self.ps.m[p_i] / self.ps.density0[p_i]) * xi


    @ti.kernel
    def compute_matrix_free_Ax_step_1(self, tmp: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            tmp_i = ti.math.vec3(0.0)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                tmp_i += (self.ps.m[p_j] * x[p_i] + self.ps.m[p_i] * x[p_j]) *  self.ps.fluid_neighbors_values[p_i, j]

            tmp[p_i] = tmp_i / self.ps.m[p_i]

    @ti.kernel
    def compute_matrix_free_Ax_step_2(self, Ax: ti.template(), z: ti.template()):

        for p_i in ti.grouped(z):
            Ax_i = 0.0
            x_i = self.ps.x[p_i]
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                Ax_i += self.ps.m[p_j] * self.nablaWij(x_i - x_j).dot(z[p_i] - z[p_j])

            Ax[p_i] = Ax_i

    # def pressure_solve(self):
    #
    #     num_iter = 0
    #     data = []
    #     for i in range(self.max_iteration):
    #
    #         self.enforce_boundary_3D(self.ps.material_fluid)
    #         self.ps.search_neighbours(self.ps.x)
    #
    #         self.precompute_values()
    #         self.compute_density()
    #
    #         # self.compute_divergence()
    #
    #         if self.toggle:
    #             avg_density_err = self.compute_source()
    #
    #             # tol = 1e-3
    #             #
    #             # # self.x.copy_from(self.ps.pressure)
    #             #
    #             # self.x.fill(0.0)
    #             # self.r.copy_from(self.b)
    #             # # add(self.r, self.b, -1.0, self.Ax)
    #             # self.p.copy_from(self.r)
    #             # rs_old = dot2(self.r, self.r)
    #             #
    #             # if rs_old > tol:
    #             #
    #             #     iter = 0
    #             #     for i in range(1000):
    #             #
    #             #         self.compute_matrix_free_Ax(self.Ap, self.p)
    #             #         alpha = rs_old / dot2(self.p, self.Ap)
    #             #         add(self.x, self.x, +alpha, self.p)
    #             #
    #             #         add(self.r, self.r, -alpha, self.Ap)
    #             #         r_norm = dot2(self.r, self.r)
    #             #
    #             #         if r_norm < tol:
    #             #             break
    #             #         rs_new = dot2(self.r, self.r)
    #             #         beta = rs_new / rs_old
    #             #         add(self.p, self.r, beta, self.p)
    #             #         rs_old = rs_new
    #             #
    #             #         iter += 1
    #             #
    #             #     print(iter)
    #
    #             self.compute_Aii()
    #
    #             # self.p.fill(0.0)
    #
    #             # self.project(self.b)
    #             self.apply_precondition(self.p, self.b)
    #             self.project(self.p)
    #
    #             iter = 0
    #             # for i in range(1000):
    #             #
    #             #     # A = D J invM J^T D
    #             #
    #             #     # step 1: Dp
    #             #     self.compute_matrix_free_Ax_step_0(self.p)
    #             #
    #             #     # step 2: (invM J^T) Dp
    #             #     self.compute_matrix_free_Ax_step_1(self.tmp, self.p)
    #             #
    #             #     # step 3: J (invM J^T) Dp
    #             #     self.compute_matrix_free_Ax_step_2(self.Ap, self.tmp)
    #             #
    #             #     # step 4: D (J invM J^T) Dp
    #             #     self.compute_matrix_free_Ax_step_0(self.Ap)
    #             #     # add(self.Ap, self.Ap, 1e-6, self.p)
    #             # #     # r = b - Ap
    #             #
    #             #     # tst = dot2(self.p, self.Ap)
    #             #     #
    #             #     # if tst < 0.0:
    #             #     #     print("fucked")
    #             #
    #             #     #r = b - Ap
    #             #     add(self.r, self.b, -1.0, self.Ap)
    #             #
    #             #     r_norm = dot2(self.r, self.r)
    #             #
    #             #     if r_norm < 0.01:
    #             #         break
    #             #
    #             #     # z = diag(A)^-1 r
    #             #     self.apply_precondition(self.z, self.r)
    #             # #
    #             # #     # p += diag(A)^-1 (b - Ap)
    #             #     add(self.p, self.p, 0.5, self.z)
    #             #     self.project(self.p)
    #             #     iter += 1
    #             # print("Jacobi iter: ", iter)
    #
    #             self.compute_matrix_free_Ax_step_0(self.p)
    #             self.compute_matrix_free_Ax_step_1(self.tmp, self.p)
    #
    #             add(self.ps.x, self.ps.x, -1.0, self.tmp)
    #
    #             # self.ps.pressure.copy_from(self.p)
    #
    #
    #             data.append(avg_density_err)
    #             # self.step_forward_x()
    #         else:
    #             avg_density_err = self.compute_lambdas_p2()
    #             data.append(avg_density_err)
    #             self.step_forward_x2()
    #
    #         num_iter += 1
    #         # if avg_density_err < 0.001:
    #         #     break
    #
    #     return num_iter

    def divergence_solve(self):

        num_iter = 0
        data = []
        for i in range(10):
            self.compute_divergence()
            avg_density_err = self.compute_lambdas_v()
            # data.append(avg_density_err)

            # self.ps.pressure.fill(0.0)
            self.step_forward_v()
            num_iter += 1
            # if avg_density_err < 0.5:
            #     break

        return num_iter

    @ti.kernel
    def mat_free_mul_D(self, Dx: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            Dx[p_i] = (self.ps.m[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])) * x[p_i]

    @ti.kernel
    def mat_free_mul_D_inv(self, D_inv_x: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            D_inv_x[p_i] = (self.ps.density[p_i] * self.ps.density[p_i] / self.ps.m[p_i]) * x[p_i]

    @ti.kernel
    def mat_free_mul_nabla_rho(self, nabla_rho_x: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            nabla_rho_x_i = 0.0
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                nabla_rho_x_i += self.ps.m[p_j] * (x[p_i] - x[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])

            nabla_rho_x[p_i] = nabla_rho_x_i
            # nabla_rho_x[p_i] = tmp_i
    

    @ti.kernel
    def compute_J_x(self, ret: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            ret_i = 0.0
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                ret_i += self.ps.m[p_j] * (x[p_i] - x[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])

            ret[p_i] = ret_i

    @ti.kernel
    def compute_J_tr_x(self, ret: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            ret_i = ti.math.vec3(0.0)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                ret_i += (self.ps.m[p_j] * x[p_i] + self.ps.m[p_i] * x[p_j]) * self.ps.fluid_neighbors_values[p_i, j]

            ret[p_i] = ret_i


    @ti.kernel
    def mat_free_mul_invM_nabla_rho_T(self, invM_nabla_rho_T_x: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            invM_nabla_rho_T_x_i = ti.math.vec3(0.0)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                invM_nabla_rho_T_x_i +=(self.ps.m[p_j] * x[p_i] + self.ps.m[p_i] * x[p_j]) * self.ps.fluid_neighbors_values[p_i, j]

            invM_nabla_rho_T_x[p_i] = invM_nabla_rho_T_x_i


    @ti.kernel
    def mat_free_mul_D_sqrt(self, out: ti.template(), x: ti.template()):
        for p_i in ti.grouped(x):
            out[p_i] = ti.sqrt(self.ps.m[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])) * x[p_i]


    @ti.kernel
    def mat_free_mul_D_sqrt_inv(self, out: ti.template(), x: ti.template()):
        for p_i in ti.grouped(x):
            out[p_i] = ti.sqrt(self.ps.density[p_i] * self.ps.density[p_i] / self.ps.m[p_i]) * x[p_i]


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

    @ti.kernel
    def measure_error_normal_equations(self) -> float:
        """
        Measure error for Normal Equations approach.
        Computes the residual norm of the pressure equation.
        """
        avg_error = 0.0
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            
            # Compute residual norm: ||D^T b - (D^T J M^(-1) J^T D) p||
            # For simplicity, we'll use the residual norm directly
            avg_error += self.r_pcg[p_i] * self.r_pcg[p_i]
        
        avg_error = ti.sqrt(avg_error / self.ps.fluid_particle_num)
        return avg_error

    @ti.kernel
    def apply_precondition_normal_equations(self, z: ti.template(), r: ti.template()):
        """
        Preconditioner for Normal Equations approach.
        Uses diagonal approximation of (D^T J M^(-1) J^T D)
        """
        for p_i in ti.grouped(r):
            # Diagonal approximation: diag(D^T J M^(-1) J^T D) ≈ D^2 * Bii
            # where Bii is the diagonal of J M^(-1) J^T and D_ii = m_i/ρ_i²
            d_val = self.ps.m[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])
            diag_approx = d_val * d_val * self.Bii[p_i]
            denom = ti.max(diag_approx, 1e-12)
            z[p_i] = r[p_i] / denom


    def pressure_solve(self):
        
        if self.method == 0:
            self.IISPH()

        

    @ti.kernel
    def compute_Hii(self, p: ti.template()):

        for p_i in ti.grouped(p):
            self.Hii[p_i] = self.Bii[p_i] + 1.0 / (p[p_i] ** 2)


    # === Matrix-free operator helpers for ADMM z-update ===
    def mat_free_Bx(self, out_scalar, x_scalar):
        # out_scalar = D_inv @ (nabla_rho @ (invM @ (nabla_rho^T @ x_scalar)))
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, x_scalar)  # tmp: vector field
        self.mat_free_mul_nabla_rho(out_scalar, self.tmp)       # out_scalar: scalar field
        # self.mat_free_mul_D_inv(out_scalar, out_scalar)         # out_scalar: scalar field

    def mat_free_B_alphaI_x(self, out_scalar, x_scalar, alpha: float):
        # out_scalar = A_tilde(x_scalar) + alpha * x_scalar
        self.mat_free_Bx(out_scalar, x_scalar)
        add(out_scalar, out_scalar, alpha, x_scalar)

    # === Matrix-free operator for Normal Equations approach ===
    def mat_free_normal_equations(self, out_scalar, x_scalar):
        # out_scalar = D^T @ (nabla_rho @ (invM @ (nabla_rho^T @ (D @ x_scalar))))
        # This implements: (D^T J M^(-1) J^T D) x
        self.mat_free_mul_D(self.tmp, x_scalar)                    # tmp = D @ x
        self.mat_free_mul_invM_nabla_rho_T(self.tmp_vec, self.tmp) # tmp_vec = M^(-1) J^T @ tmp
        self.mat_free_mul_nabla_rho(self.tmp2, self.tmp_vec)        # tmp2 = J @ tmp_vec
        self.mat_free_mul_D(out_scalar, self.tmp2)                  # out_scalar = D @ tmp2


    # @ti.kernel
    # def compute_Mii(self, alpha: float):
    #     # ADMM preconditioner
    #     for p_i in range(self.ps.fluid_particle_num):
    #         # Jacobi preconditioner diag(M) where M = A_tilde + alpha*I
    #         # A_tilde diag ≈ D_inv * diag(nabla_rho invM nabla_rho^T) ≈ (density/m) * Bii
    #         density_over_mass = self.ps.density[p_i] / self.ps.m[p_i]
    #         self.Hii[p_i] = ti.max(density_over_mass * self.Bii[p_i] + alpha, 1e-12)


    @ti.kernel
    def clamp_velocity(self, v: ti.template(), vmax: float):
        for p_i in ti.grouped(v):
            vi = v[p_i]
            n = vi.norm()
            if n > vmax:
                v[p_i] = (vmax / (n + 1e-12)) * vi


    def Barrier(self):

        # for i in range(10):
        self.PCG()
        self.project(self.y)
        # self.mat_free_mul_D(self.y, self.p)
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.y)
        add(self.v_tmp, self.ps.v, -self.dt, self.tmp)


        # print("TODO")


    def PCG(self):

        eps = 1e-1
        pcgIter = 0
        # self.mat_free_mul_D(self.y, self.p)
        self.x.copy_from(self.y)
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.x)
        self.mat_free_mul_nabla_rho(self.Ap_b, self.tmp)
        add(self.r_pcg, self.b, -1.0, self.Ap_b)

        self.apply_precondition(self.z_pcg, self.Bii, self.r_pcg)


        self.p_pcg.copy_from(self.z_pcg)
        rz_old = dot2(self.r_pcg, self.z_pcg)
        rr = dot2(self.r_pcg, self.r_pcg)

        if rr < eps:
            # self.y.copy_from(self.x)
            return

        pcgIter += 1

        for i in range(1000):

            # compute Ap with matrix-free fashion
            self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.p_pcg)
            self.mat_free_mul_nabla_rho(self.Ap_b, self.tmp)

            pAp = dot2(self.p_pcg, self.Ap_b)
            # print(pAp)
            alpha = rz_old / pAp
            add(self.x, self.x, +alpha, self.p_pcg)
            add(self.r_pcg, self.r_pcg, -alpha, self.Ap_b)

            self.apply_precondition(self.z_pcg, self.Bii, self.r_pcg)
            rz_new = dot2(self.r_pcg, self.z_pcg)
            rr = dot2(self.r_pcg, self.r_pcg)

            # print(rs_new)

            if rr < eps:
                # self.y.copy_from(self.x)
                break

            pcgIter += 1
            beta = rz_new / rz_old
            add(self.p_pcg, self.z_pcg, beta, self.p_pcg)
            rz_old = rz_new

        self.y.copy_from(self.x)

        print(pcgIter)
        # self.mat_free_mul_D(self.p, self.y)

    @ti.kernel
    def compute_b_admm(self, alpha: float):

        for i in ti.grouped(self.b_admm):
            self.b_admm[i] = self.b[i] + alpha * (self.z_admm[i] - self.u[i])

    def update_y(self, alpha_admm):

        #compute b_admm = b + alpha_admm * (z - u)
        self.compute_b_admm(alpha_admm)

        # self.Hii.fill(0.0)
        # add(self.Hii, self.Hii, alpha_admm, self.i)

        eps = 50
        # self.mat_free_mul_D(self.y, self.p)

        # self.y.fill(0.0)
        self.x.copy_from(self.y)
        self.mat_free_B_alphaI_x(self.Ap_b, self.x, alpha_admm)
        add(self.r_pcg, self.b_admm, -1.0, self.Ap_b)
        # self.r_pcg.copy_from(self.b_admm)
        self.apply_precondition(self.z_pcg, self.Hii, self.r_pcg)

        self.p_pcg.copy_from(self.z_pcg)
        rz_old = dot2(self.r_pcg, self.z_pcg)
        rr = sqrt(dot2(self.r_pcg, self.r_pcg))

        if rr < eps:
            # self.y.copy_from(self.x)
            return

        # self.stats_pcg_iter += 1
        pcgIter = 0
        for i in range(1):

            # compute Ap with matrix-free fashion
            self.mat_free_B_alphaI_x(self.Ap_b, self.p_pcg, alpha_admm)

            # self.Ap_b.fill(0.0)
            # add(self.Ap_b, self.Ap_b, alpha_admm, self.p_pcg)

            pAp = dot2(self.p_pcg, self.Ap_b)
            # print(pAp)
            alpha = rz_old / pAp
            add(self.x, self.x, +alpha, self.p_pcg)

            # self.tmp.fill(0.0)
            # self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.z_admm)
            # add(self.v_tmp, self.ps.v, -self.dt, self.tmp)
            #
            # err = self.measure_error(self.v_tmp)
            # if  err < eps:
            #     break

            add(self.r_pcg, self.r_pcg, -alpha, self.Ap_b)
            self.apply_precondition(self.z_pcg, self.Hii, self.r_pcg)
            rz_new = dot2(self.r_pcg, self.z_pcg)
            rr = sqrt(dot2(self.r_pcg, self.r_pcg))

            # print(rr)
            # print(rs_new)

            if rr < eps:
                # self.y.copy_from(self.x)
                break

            self.stats_pcg_iter += 1
            pcgIter += 1
            beta = rz_new / rz_old
            add(self.p_pcg, self.z_pcg, beta, self.p_pcg)
            rz_old = rz_new

        self.y.copy_from(self.x)
        # print("PCG iter: ", pcgIter)


    def update_z(self):

        add(self.z_admm, self.y, 1.0, self.u)
        self.project(self.z_admm)

    def update_u(self):

        add(self.r_admm, self.y, -1.0, self.z_admm)
        add(self.u, self.u, 1.0, self.r_admm)


    def ADMM(self):

        iter = 0
        tol = pow(10, -self.tol)

        self.stats_pcg_iter = 0
        alpha = 1e-2 * sqrt(dot2(self.Bii, self.Bii))
        alpha = 0.0
        inner_pcg_iters = 5

        # ADMM variables initialization

        # self.compute_Mii(alpha)
        add(self.Hii, self.Bii, alpha, self.i)

        self.z_admm.fill(0.0)  # primal variable z
        self.w.fill(0.0)  # consensus variable w
        self.u.fill(0.0)  # dual variable u

        for i in range(100):
            # print("iter: ", i)
            self.update_y(alpha_admm=alpha)
            self.update_z()
            # self.update_u()

            self.tmp.fill(0.0)
            self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.z_admm)
            add(self.v_tmp, self.ps.v, -self.dt, self.tmp)
            err = self.measure_error(self.v_tmp)
            if err < tol  and self.stats_iter > 2:
               break
            self.stats_iter += 1

        # print("PCG iteration: ", self.stats_pcg_iter)
        print("ADMM iteration: ", self.stats_iter)
        
        # Log iteration data if logging is enabled
        if self.enable_logging:
            self.iteration_log.append([self.current_frame, f"ADMM", self.stats_iter])
        
        # Apply final solution to velocity: v <- v - dt * (invM nabla_rho^T w)
        # self.tmp.fill(0.0)
        # self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.z_admm)
        # add(self.v_tmp, self.ps.v, -self.dt, self.tmp)

        # self.clamp_velocity(self.v_tmp, vmax=3.0)
        self.ps.v.copy_from(self.v_tmp)


    def _apply_matrix_type0(self, out, solution):
        """Apply B matrix: J M^-1 J^T"""
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, solution)
        self.mat_free_mul_nabla_rho(out, self.tmp)

    def _apply_matrix_type1(self, out, solution):
        """Apply √D B √D matrix: √D J M^-1 J^T √D"""
        # √D * solution
        self.mat_free_mul_D_sqrt(self.tmp2, solution)
        # J M^-1 J^T (√D * solution)
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.tmp2)
        self.mat_free_mul_nabla_rho(self.tmp2, self.tmp)
        # √D * result
        self.mat_free_mul_D_sqrt(out, self.tmp2)

    def _apply_matrix_type2(self, out, solution):
        """Apply D B D matrix: D J M^-1 J^T D"""
        # D * solution
        self.mat_free_mul_D(self.tmp2, solution)
        # J M^-1 J^T (D * solution)
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.tmp2)
        self.mat_free_mul_nabla_rho(self.tmp2, self.tmp)
        # D * result
        self.mat_free_mul_D(out, self.tmp2)

    def _apply_matrix_type3(self, out, solution):
        """Apply non-symmetric A matrix: J M^-1 J^T D"""
        # This is the baseline non-symmetric matrix
        # J M^-1 J^T D * solution
        self.mat_free_mul_D(self.tmp2, solution)  # tmp2 = D * solution
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.tmp2)  # tmp = M^-1 J^T D * solution
        self.mat_free_mul_nabla_rho(out, self.tmp)  # out = J M^-1 J^T D * solution

    @ti.kernel
    def _apply_preconditioner_type1(self, z: ti.template(), r: ti.template()):
        """Preconditioner for √D B √D: use √D * Bii * √D scaling"""
        for p_i in ti.grouped(r):
            # Diagonal approximation: √D * diag(B) * √D = (m/density²) * Bii
            diag_approx = (self.ps.m[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])) * self.Bii[p_i]
            denom = ti.max(diag_approx, 1e-12)
            z[p_i] = r[p_i] / denom

    @ti.kernel 
    def _apply_preconditioner_type2(self, z: ti.template(), r: ti.template()):
        """Preconditioner for D B D: use D * Bii * D scaling"""
        for p_i in ti.grouped(r):
            # Diagonal approximation: D * diag(B) * D = (m/density²)^2 * Bii
            d_val = self.ps.m[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])
            diag_approx = d_val * d_val * self.Bii[p_i]
            denom = ti.max(diag_approx, 1e-12)
            z[p_i] = r[p_i] / denom

    @ti.kernel
    def _apply_preconditioner_type3(self, z: ti.template(), r: ti.template()):
        """Preconditioner for non-symmetric A: use diagonal of J M^-1 J^T D"""
        for p_i in ti.grouped(r):
            # Diagonal approximation: diag(J M^-1 J^T D) = D * diag(J M^-1 J^T) = (m/density²) * Bii
            diag_approx = (self.ps.m[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])) * self.Bii[p_i]
            denom = ti.max(diag_approx, 1e-12)
            z[p_i] = r[p_i] / denom


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

        self.ps.v.copy_from(self.ps.v_adv)
        self.compute_Aii(self.iisph_vanilla)
        self.compute_b(self.b, self.ps.v, self.dt)

        tol = pow(10, -self.tol)
        self.p.fill(0.0)
        iter = 0
        relax = 0.5
        for _ in range(self.max_iteration):
            

            # if self.iisph_vanilla:
            #     coef_wise_op(self.p, self.p, self.Dii, 0)

            # self.compute_J_tr_x(self.tmp, self.p)
            # coef_wise_op(self.tmp, self.tmp, self.ps.m, 1)

            if self.iisph_vanilla:
                # self.compute_invM_J_tr_D_x(self.tmp, self.p)
                # coef_wise_op(self.p, self.p, self.Dii, 0)
                
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

    
            add(self.p, self.p, relax, self.dp)  # Initialize p with b
            max(self.p)  # Ensure non-negativity

    def NormalEquations(self):
        """
        Normal Equations approach: solve (D^T J M^(-1) J^T D) p = D^T b
        This creates a symmetric system by multiplying both sides by D^T
        """
        # Compute D^T b for the right-hand side
        self.mat_free_mul_D(self.b_normal, self.b)
        
        # Initialize solution vector
        self.p.fill(0.0)
        
        tol = pow(10, -self.tol)
        for i in range(self.max_iteration):
            # Compute Ap = (D^T J M^(-1) J^T D) p
            self.mat_free_normal_equations(self.Ap, self.p)
            
            # Compute residual: r = D^T b - Ap
            add(self.r_pcg, self.b_normal, -1.0, self.Ap)
            
            # Check convergence
            err = self.measure_error_normal_equations()
            if err < tol and self.stats_iter > 2:
                break
                
            self.stats_iter += 1
            
            # Jacobi update: p = p + D^(-1) r (using diagonal approximation)
            self.apply_precondition_normal_equations(self.z_pcg, self.r_pcg)
            add(self.p, self.p, 1.0, self.z_pcg)
            
            # Apply projection to ensure non-negative pressure
            self.project(self.p)
        
        print("Normal Equations iteration: ", self.stats_iter)
        
        # Log iteration data if logging is enabled
        if self.enable_logging:
            self.iteration_log.append([self.current_frame, "NormalEq", self.stats_iter])
        
        # Apply the solution to velocity: v = v - dt * (M^(-1) J^T D p)
        self.mat_free_mul_D(self.tmp, self.p)  # tmp = D p
        self.mat_free_mul_invM_nabla_rho_T(self.tmp_vec, self.tmp)  # tmp_vec = M^(-1) J^T D p
        add(self.v_tmp, self.ps.v, -self.dt, self.tmp_vec)
        self.ps.v.copy_from(self.v_tmp)

    def save_iteration_logs(self):
        """Save iteration logs to files in log folder"""
        if not self.iteration_log:
            print("No iteration data to save")
            return
            
        # Create log directory if it doesn't exist
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Group data by matrix type
        data_by_type = {}
        for frame, matrix_type, iterations in self.iteration_log:
            if matrix_type not in data_by_type:
                data_by_type[matrix_type] = []
            data_by_type[matrix_type].append([frame, iterations])
        
        # Save each matrix type to separate file
        for matrix_type, data in data_by_type.items():
            filename = f"iterations_type_{matrix_type}.json"
            filepath = os.path.join(log_dir, filename)
            
            log_data = {
                "matrix_type": matrix_type,
                "data": data,  # [[frame, iterations], ...]
                "total_frames": len(data),
                "avg_iterations": sum(row[1] for row in data) / len(data) if data else 0
            }
            
            with open(filepath, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"Saved {len(data)} iteration records for type {matrix_type} to {filepath}")
        
        # Also save combined data
        combined_filepath = os.path.join(log_dir, "iterations_combined.json")
        combined_data = {
            "all_data": self.iteration_log,  # [[frame, matrix_type, iterations], ...]
            "summary": {
                matrix_type: {
                    "count": len(data),
                    "avg_iterations": sum(row[1] for row in data) / len(data) if data else 0
                }
                for matrix_type, data in data_by_type.items()
            }
        }
        
        with open(combined_filepath, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"Saved combined iteration data to {combined_filepath}")

    def reset_logging(self):
        """Reset logging system and save current data"""
        if self.iteration_log and self.enable_logging:
            self.save_iteration_logs()
        
        # Reset the log
        self.iteration_log = []
        self.current_frame = 0
        print("Iteration logging reset")
        
        # Reset spectral radius logging too
        if self.enable_spectral_analysis and self.spectral_radius_analyzer:
            self.spectral_radius_analyzer.reset_logging()
    
    def save_spectral_radius_data(self, filename="spectral_radius_log.json"):
        """Save spectral radius analysis data"""
        if self.enable_spectral_analysis and self.spectral_radius_analyzer:
            self.spectral_radius_analyzer.save_spectral_radius_data(filename)
        else:
            print("Spectral radius analysis not enabled or available")
    
    def plot_spectral_radius(self, save_path="plots/spectral_radius.png"):
        """Plot spectral radius analysis"""
        if self.enable_spectral_analysis and self.spectral_radius_analyzer:
            self.spectral_radius_analyzer.plot_spectral_radius(save_path)
        else:
            print("Spectral radius analysis not enabled or available")
    
    def print_spectral_radius_summary(self):
        """Print spectral radius analysis summary"""
        if self.enable_spectral_analysis and self.spectral_radius_analyzer:
            self.spectral_radius_analyzer.print_summary()
        else:
            print("Spectral radius analysis not enabled or available")

    def increment_frame(self):
        """Increment frame counter for logging"""
        self.current_frame += 1

    def substep(self):
        
        # print("test")
        # Increment frame counter for logging
        # if self.enable_logging:
        #     self.increment_frame()

        self.ps.search_neighbours(self.ps.x)
        # self.ps.x_old.copy_from(self.ps.x)
        self.compute_non_pressure_forces()
        self.compute_density()
        self.precompute_values()
        self.advect_velocity(self.dt)
        self.pressure_solve()
        self.advect_position(self.dt)
