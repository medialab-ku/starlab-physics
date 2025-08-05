import taichi as ti
from sph_base import SPHBase


class PBF3Solver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        self.exponent = 7.0
        self.exponent = self.ps.cfg.get_cfg("exponent")

        self.stiffness = 50000.0
        self.stiffness = self.ps.cfg.get_cfg("stiffness")

        self.surface_tension = 0.00
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")

        self.lda = self.ps.pressure

        print("method: PBF3")

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())

    @ti.kernel
    def compute_densities(self):
        # for p_i in range(self.ps.particle_num[None]):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            self.ps.density[p_i] *= self.density_0


    @ti.func
    def compute_lambdas_task(self, p_i, p_j, ret: ti.template()):

        # schur = 0.0
        dc_dxi = ti.math.vec3(0.0)
        x_i = self.ps.x[p_i]
        # Fluid neighbors
        dc_drho_i = (self.density_0 * self.ps.m_V[p_i]) / (self.ps.density[p_i] * self.ps.density[p_i])
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            m_j = self.density_0 * self.ps.m_V[p_j]
            nabla_cij = dc_drho_i * m_j * self.spiky_kernel_derivative(x_i - x_j)
            # dc_dxi -= nabla_cij
            ret[3] += nabla_cij.dot(nabla_cij) / m_j

            for i in range(3):
                ret[i] -= nabla_cij[i]
            # Compute the pressure force contribution, Symmetric Formula
            # ret += -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) * self.cubic_kernel_derivative(x_i - x_j)

        # ret = schur
        # for i in range(3):
        #     ret[i] = dc_dxi[i]
        #
        # ret[3] = schur

    @ti.func
    def compute_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        density_i = self.ps.density[p_i]
        lambda_i = self.ps.pressure[p_i]
        # Fluid neighbors

        scorr = 0.0
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            m_j = self.density_0 * self.ps.m_V[p_j]
            density_j = self.ps.density[p_j]  # TODO: The density_0 of the neighbor may be different when the fluid density is different
            lambda_j = self.ps.pressure[p_j]
            # Compute the pressure force contribution, Symmetric Formula
            ret += m_j * (lambda_i / density_i ** 2 + lambda_j / density_j ** 2) * self.spiky_kernel_derivative(x_i - x_j)
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
            self.ps.acceleration[p_i] = d_v
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[p_i] = d_v

    @ti.kernel
    def advect(self):
        # Symplectic Euler
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.x_old[p_i] = self.ps.x[p_i]
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                # self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]


    @ti.kernel
    def compute_lambdas(self) -> float:

        eps = 1e-6
        avg_density_err = 0.0

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            m_i = (self.density_0 * self.ps.m_V[p_i])
            c = (m_i / self.density_0) * ti.min(self.density_0 / self.ps.density[p_i] -  1.0, 0.0)
            avg_density_err += c
            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])
            # ret = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_lambdas_task, ret)

            schur = ret[3]
            dc_dxi = ti.Vector([ret[0], ret[1], ret[2]])
            schur += (dc_dxi.dot(dc_dxi)) / m_i
            self.ps.pressure[p_i] = -c / (schur + eps)

        avg_density_err /= self.ps.fluid_particle_num
        return avg_density_err

    @ti.kernel
    def update_positions(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0)
                continue
            elif self.ps.is_dynamic_rigid_body(p_i):
                continue

            m_i = (self.density_0 * self.ps.m_V[p_i])
            dx = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dx)
            self.ps.x[p_i] -= dx




    @ti.kernel
    def update_velocities(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] = (self.ps.x[p_i] - self.ps.x_old[p_i])/ self.dt[None]


    def substep(self):

        self.compute_non_pressure_forces()
        self.compute_densities()

        self.advect()

        # self.enforce_boundary_3D(self.ps.material_fluid)
        # self.ps.initialize_particle_system()
        num_iter = 0
        data = []
        for i in range(20):

            self.enforce_boundary_3D(self.ps.material_fluid)
            self.compute_densities()
            avg_density_err = self.compute_lambdas()
            data.append(avg_density_err)
            self.update_positions()
            num_iter += 1
            # if avg_density_err < 0.001:
            #     break

        # if num_iter > 61:
        #
        #     for i in data:
        #         print(i)

        # print("num iteration: ", num_iter)
            # self.enforce_boundary_3D(self.ps.material_fluid)


        # self.update_velocities()


        # self.compute_non_pressure_forces()
        # self.compute_pressure_forces()
