import taichi as ti
import math
from sph_base import SPHBase
from math_utils import *


class PBF2Solver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        # self.exponent = 7.0
        # self.exponent = self.ps.cfg.get_cfg("exponent")
        #
        # self.stiffness = 50000.0
        # self.stiffness = self.ps.cfg.get_cfg("stiffness")

        self.surface_tension = 0.0
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")

        self.nablaWij = self.cubic_kernel_derivative
        self.lda = self.ps.pressure
        self.method = 0

        self.tol = 2
        self.toggle = True
        self.max_iteration = 1000

        self.div = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)

        self.v_tmp = ti.Vector.field(n=3, dtype=float, shape=self.ps.fluid_particle_num)

        self.Aii = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.Bii = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.Hii = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.Ap  = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.x   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.p   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.y   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.b   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.z   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.r   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)

        # ADMM variables
        self.b_tilde = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.w = ti.field(dtype=float, shape=self.ps.fluid_particle_num)  # consensus variable
        self.w_prev = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.u = ti.field(dtype=float, shape=self.ps.fluid_particle_num)  # dual variable


        self.grad = ti.field(dtype=float, shape=self.ps.fluid_particle_num)

        self.Ap_b   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        # self.x   = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.p_b    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.y_b    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.b_b    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.z_b    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.r_b    = ti.field(dtype=float, shape=self.ps.fluid_particle_num)
        self.grad_b = ti.field(dtype=float, shape=self.ps.fluid_particle_num)

        print("method: PBF2")




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
    def advect_velocity(self):
        # Symplectic Euler
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                # self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]

    @ti.kernel
    def advect_position(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                # self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]


    @ti.func
    def compute_lambdas_task(self, p_i, p_j, ret: ti.template()):

        # schur = 0.0
        m_i = (self.density_0 * self.ps.m_V[p_i])
        # dc_dxi = ti.math.vec3(0.0)
        x_i = self.ps.x[p_i]
        # Fluid neighbors
        # dc_drho_i = self.density_0 * self.ps.m_V[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            # m_j = self.density_0 * self.ps.m_V[p_j]
            nabla_cij = (self.ps.m[p_i] / self.ps.density0[p_i]) * self.ps.m[p_j] * self.nablaWij(x_i - x_j)
            # dc_dxi -= nabla_cij
            ret[3] += nabla_cij.dot(nabla_cij) / self.ps.m[p_j]

            for i in range(3):
                ret[i] -= nabla_cij[i]

    @ti.kernel
    def compute_source(self) -> float:

        eps = 1e-6
        avg_density_err = 0.0

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            self.b[p_i] = (self.ps.m[p_i] / self.ps.density0[p_i]) * (self.ps.density[p_i] - self.ps.density0[p_i])
            avg_density_err += (ti.max(self.b[p_i], 0.0) / self.ps.m[p_i])

        avg_density_err /= self.ps.fluid_particle_num
        return avg_density_err

    @ti.func
    def compute_Ax_task(self, p_i, p_j, ret: ti.template()):

        # schur = 0.0
        m_i = (self.density_0 * self.ps.m_V[p_i])
        # dc_dxi = ti.math.vec3(0.0)
        x_i = self.ps.x[p_i]
        # Fluid neighbors
        # dc_drho_i = self.density_0 * self.ps.m_V[p_i] / (self.ps.density[p_i] * self.ps.density[p_i])
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            # m_j = self.density_0 * self.ps.m_V[p_j]
            nabla_cij = (self.ps.m[p_i] / self.ps.density0[p_i]) * self.ps.m[p_j] * self.nablaWij(x_i - x_j)
            # dc_dxi -= nabla_cij
            ret[3] += nabla_cij.dot(nabla_cij) / self.ps.m[p_j]

            for i in range(3):
                ret[i] -= nabla_cij[i]

    @ti.kernel
    def computeAx(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])
            self.ps.for_all_neighbors(p_i, self.computeAx, ret)


    @ti.kernel
    def compute_Aii(self):

        eps = 1e-3

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            Bii = 0.0
            dc_dxi = ti.math.vec3(0.0)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                nabla_cij = self.ps.m[p_j] * self.ps.fluid_neighbors_values[p_i, j]
                Bii += nabla_cij.dot(nabla_cij) / self.ps.m[p_j]

                dc_dxi -= nabla_cij
            Bii += dc_dxi.dot(dc_dxi) / self.ps.m[p_i]

            self.Bii[p_i] = Bii + eps
            self.Aii[p_i] = (self.ps.m[p_i] / self.ps.density[p_i]) * Bii + eps


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
                self.ps.v[p_i] = (self.ps.x[p_i] - self.ps.x_old[p_i])/ self.dt[None]

    @ti.kernel
    def project(self, p: ti.template()):
        for p_i in ti.grouped(p):
            p[p_i] = ti.max(p[p_i], 0.0)

    @ti.kernel
    def update_pressure_acceleration(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                v_tmp = (self.ps.x[p_i] - self.ps.x_old[p_i]) / self.dt[None]
                self.ps.acceleration[p_i] += (v_tmp - self.ps.v_old[p_i]) / self.dt[None]


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
            Dx[p_i] = (self.ps.m[p_i] / self.ps.density[p_i]) * x[p_i]

    @ti.kernel
    def mat_free_mul_D_inv(self, D_inv_x: ti.template(), x: ti.template()):

        for p_i in ti.grouped(x):
            D_inv_x[p_i] = (self.ps.density[p_i] / self.ps.m[p_i]) * x[p_i]

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
            out[p_i] = ti.sqrt(self.ps.m[p_i] / self.ps.density[p_i]) * x[p_i]


    @ti.kernel
    def mat_free_mul_D_sqrt_inv(self, out: ti.template(), x: ti.template()):
        for p_i in ti.grouped(x):
            out[p_i] = ti.sqrt(self.ps.density[p_i] / self.ps.m[p_i]) * x[p_i]


    @ti.kernel
    def compute_b(self, b: ti.template(), v: ti.template()):

        dtSq = self.dt[None] ** 2
        for p_i in ti.grouped(b):
            div_i = 0.0
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                div_i += self.ps.m[p_j] * (v[p_i] - v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])

            b[p_i] = (self.ps.density[p_i] + self.dt[None] * div_i - self.ps.density0[p_i]) / dtSq

    @ti.kernel
    def measure_error(self, v: ti.template()) -> float:

        avg_error = 0.0
        for p_i in ti.grouped(v):
            div_i = 0.0
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                div_i += self.ps.m[p_j] * (v[p_i] - v[p_j]).dot(self.ps.fluid_neighbors_values[p_i, j])

            avg_error = ti.max(self.ps.density[p_i] + self.dt[None] * div_i - self.ps.density0[p_i], 0.0) / self.ps.density0[p_i]

        return avg_error




    def pressure_solve(self):

        self.compute_Aii()
        self.p.fill(0.0)
        self.y.fill(0.0)
        self.compute_b(self.b, self.ps.v)

        # method = 0
        if self.method == 0:
            self.ProjectedJacobi()
        elif self.method == 1:
            self.ADMM()
        elif self.method == 2:
            # print("test")
            self.Barrier()

    @ti.kernel
    def compute_Hii(self, p: ti.template()):

        for p_i in ti.grouped(p):
            self.Hii[p_i] = self.Bii[p_i] + 1.0 / (p[p_i] ** 2)


    # === Matrix-free operator helpers for ADMM z-update ===
    def apply_Atilde(self, out_scalar, x_scalar):
        # out_scalar = D_inv @ (nabla_rho @ (invM @ (nabla_rho^T @ x_scalar)))
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, x_scalar)  # tmp: vector field
        self.mat_free_mul_nabla_rho(out_scalar, self.tmp)       # out_scalar: scalar field
        self.mat_free_mul_D_inv(out_scalar, out_scalar)         # out_scalar: scalar field

    def apply_Atilde_plus_alphaI(self, out_scalar, x_scalar, alpha: float):
        # out_scalar = A_tilde(x_scalar) + alpha * x_scalar
        self.apply_Atilde(out_scalar, x_scalar)
        add(out_scalar, out_scalar, alpha, x_scalar)


    @ti.kernel
    def compute_Mii(self, alpha: float):
        # ADMM preconditioner
        for p_i in range(self.ps.fluid_particle_num):
            # Jacobi preconditioner diag(M) where M = A_tilde + alpha*I
            # A_tilde diag ≈ D_inv * diag(nabla_rho invM nabla_rho^T) ≈ (density/m) * Bii
            density_over_mass = self.ps.density[p_i] / self.ps.m[p_i]
            self.Hii[p_i] = ti.max(density_over_mass * self.Bii[p_i] + alpha, 1e-12)


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
        add(self.v_tmp, self.ps.v, -self.dt[None], self.tmp)


        # print("TODO")


    def PCG(self):

        eps = 1e-1
        pcgIter = 0
        # self.mat_free_mul_D(self.y, self.p)
        self.x.copy_from(self.y)
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.x)
        self.mat_free_mul_nabla_rho(self.Ap_b, self.tmp)
        add(self.r, self.b, -1.0, self.Ap_b)

        self.apply_precondition(self.z, self.Bii, self.r)


        self.p_b.copy_from(self.z)
        rz_old = dot2(self.r, self.z)
        rr = dot2(self.r, self.r)

        if rr < eps:
            # self.y.copy_from(self.x)
            return

        pcgIter += 1

        for i in range(1000):

            # compute Ap with matrix-free fashion
            self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.p_b)
            self.mat_free_mul_nabla_rho(self.Ap_b, self.tmp)

            pAp = dot2(self.p_b, self.Ap_b)
            # print(pAp)
            alpha = rz_old / pAp
            add(self.x, self.x, +alpha, self.p_b)
            add(self.r, self.r, -alpha, self.Ap_b)

            self.apply_precondition(self.z, self.Bii, self.r)
            rz_new = dot2(self.r, self.z)
            rr = dot2(self.r, self.r)

            # print(rs_new)

            if rr < eps:
                # self.y.copy_from(self.x)
                break

            pcgIter += 1
            beta = rz_new / rz_old
            add(self.p_b, self.z, beta, self.p_b)
            rz_old = rz_new

        self.y.copy_from(self.x)

        print(pcgIter)
        # self.mat_free_mul_D(self.p, self.y)


    def ADMM(self):

        iter = 0
        tol = pow(10, -self.tol)
        alpha = 0.2
        inner_pcg_iters = 5

        # ADMM variables initialization
        self.z.fill(0.0)  # primal variable z
        self.w.fill(0.0)  # consensus variable w
        self.u.fill(0.0)  # dual variable u

        # z = D * p
        self.mat_free_mul_D(self.z, self.p)

        # b -> b_tilde: D_inv * b
        self.compute_b(self.b, self.ps.v)
        self.mat_free_mul_D_inv(self.b_tilde, self.b)  # b_tilde = D_inv * b

        # Matrix for the z-update step: M = A_tilde + alpha*I

        for i in range(self.max_iteration):
            # Initialize tmp at the beginning of each iteration
            self.tmp.fill(0.0)

            # === z-update: Solve (A_tilde + alpha*I)z = b_tilde + alpha*(w - u) ===
            # Compute RHS: r = b_tilde + alpha * (w - u)
            add(self.r, self.b_tilde, alpha, self.w)      # r = b_tilde + alpha * w
            add(self.r, self.r, -alpha, self.u)           # r = r - alpha * u

            # ================= Inner PCG for (A_tilde + alpha I) z = r =================
            # PCG preparation
            # Jacobi preconditioner diag(M) and solve
            self.compute_Mii(alpha)

            pcg_eps = 1e-1
            # Use current z as initial guess then back to z
            self.x.copy_from(self.z)
            
            self.apply_Atilde_plus_alphaI(self.Ap_b, self.x, alpha)
            add(self.r_b, self.r, -1.0, self.Ap_b)
            
            self.apply_precondition(self.z_b, self.Hii, self.r_b)
            self.p_b.copy_from(self.z_b)
            rz_old = dot2(self.r_b, self.z_b)
            rr = dot2(self.r_b, self.r_b)

            # PCG iteration
            for j in range(inner_pcg_iters):
                self.apply_Atilde_plus_alphaI(self.Ap_b, self.p_b, alpha)
                pAp = dot2(self.p_b, self.Ap_b)

                # if pAp < 1e-12:
                #     break
                    
                pcg_alpha = rz_old / pAp
                add(self.x, self.x, pcg_alpha, self.p_b)
                add(self.r_b, self.r_b, -pcg_alpha, self.Ap_b)
                
                self.apply_precondition(self.z_b, self.Hii, self.r_b)
                rz_new = dot2(self.r_b, self.z_b)
                rr = dot2(self.r_b, self.r_b)
                
                if rr < pcg_eps:
                    break
                    
                pcg_beta = rz_new / rz_old
                add(self.p_b, self.z_b, pcg_beta, self.p_b)
                rz_old = rz_new
            
            self.z.copy_from(self.x)

            # === w-update: Projection w = max(0, z + u) ===
            add(self.w, self.z, 1.0, self.u)  # w = z + u
            self.project(self.w)                 # w = max(0, z + u)

            # === u-update: Dual variable update u = u + (z - w) ===
            add(self.r_b, self.z, -1.0, self.w)  # r_b = z - w (scalar field)
            add(self.u, self.u, 1.0, self.r_b)   # u = u + r_b

            self.mat_free_mul_D_inv(self.p, self.w)

            # Compute velocity candidate v_tmp = v - dt * (invM nabla_rho^T w)
            self.tmp.fill(0.0)
            self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.p)
            add(self.v_tmp, self.ps.v, -self.dt[None], self.tmp)

            err = self.measure_error(self.v_tmp)

            if err < tol and iter > 2:
                break
            iter += 1

        # Apply final solution to velocity: v <- v - dt * (invM nabla_rho^T w)
        self.tmp.fill(0.0)
        self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.p)
        add(self.v_tmp, self.ps.v, -self.dt[None], self.tmp)

        # self.clamp_velocity(self.v_tmp, vmax=3.0)
        self.ps.v.copy_from(self.v_tmp)


    def ProjectedJacobi(self):

        iter = 0
        tol = pow(10, -self.tol)
        for i in range(self.max_iteration):

            # Ap = nabla rho invM nabla rhoT Dp

            # if self.toggle is False:
            self.mat_free_mul_D(self.y, self.p)

            self.mat_free_mul_invM_nabla_rho_T(self.tmp, self.y)
            add(self.v_tmp, self.ps.v, -self.dt[None], self.tmp)
            err = self.measure_error(self.v_tmp)

            if err < tol and iter > 2:
                break

            iter += 1
            self.mat_free_mul_nabla_rho(self.Ap, self.tmp)

            # z = omega * diag(A)^-1 (r - Ap)
            add(self.r, self.b, -1.0, self.Ap)

            # if self.toggle:
            #     self.apply_precondition(self.z, self.Bii, self.r)
            #     add(self.y, self.y, 1.0, self.z)
            #     # p = max(p + z, 0.0)
            #     self.project(self.y)
            # else:
            self.apply_precondition(self.z, self.Aii, self.r)
            add(self.p, self.p, 1.0, self.z)
            # p = max(p + z, 0.0)
            self.project(self.p)

        self.ps.v.copy_from(self.v_tmp)

    def substep(self):

        self.ps.search_neighbours(self.ps.x)
        # self.ps.x_old.copy_from(self.ps.x)
        self.compute_non_pressure_forces()
        self.compute_density()
        self.precompute_values()
        self.advect_velocity()
        self.pressure_solve()

        self.advect_position()
