import taichi as ti
import math
import os
import json
import time
import numpy as np
# from sph_base import SPHBase
from math_utils import *


@ti.data_oriented
class Framework:
    def __init__(self, particle_system, neighbour_search, pressure):
        # super().__init__(particle_system)

        self.ps = particle_system
        self.ns = neighbour_search
        self.pressure = pressure

        self.dt = self.ps.cfg.get_cfg("timeStepSize")
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity

        self.viscosity = 0.01  # viscosity
        self.surface_tension = 0.01
        self.adhesion_coeff = self.surface_tension
        self.time = 0.0
        self.nablaWij = self.spiky_kernel_derivative
        self.Wij = self.cubic_kernel

    # -----------------------------
    # Stats helpers
    # -----------------------------

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

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
                r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(
            r)
        return res

    @ti.func
    def cohesion_term(self, r):
        h = self.ps.support_radius
        rn = r.norm()
        res = 0.0
        k = 32.0 / (np.pi * h ** 9)
        if rn <= 0.0 or rn > h:
            res = 0.0

        elif rn > 0.5 * h:
            t = (h - rn) * rn
            res = k * (t * t * t)
        else:
            t = 2.0 * (h - rn) * rn
            res = k * ((t * t * t) - (h ** 6) / 64.0)

        return res

    @ti.func
    def adhesion_term(self, r):
        h = self.ps.support_radius
        rn = r.norm()
        k = 0.007 / (h ** 3.25)
        arg = -4.0 * rn * rn / h + 6.0 * rn - 2.0 * h
        arg = ti.max(arg, 0.0)
        res = 0.0
        if rn <= h and rn > 0.5 * h:
            res = k * ti.sqrt(ti.sqrt(arg))

        return res


    @ti.kernel
    def initialize_boundary_neighbors(self):
        for p_i in ti.grouped(self.ps.x):
            sum_Wij = 0.0
            # Condition for boundary particles
            if self.ps.material[p_i] == self.ps.material_solid:

                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    if self.ps.material[p_j] != self.ps.material_solid:
                        continue

                    if self.ps.object_id[p_j] != self.ps.object_id[p_i]:
                        continue

                    sum_Wij += self.Wij((self.ps.x[p_i] - self.ps.x[p_j]).norm())

                if sum_Wij > 1e-12:
                    self.ps.m[p_i] = 1.5 * self.ps.density0[p_i] / sum_Wij
                    self.ps.m_V[p_i] = self.ps.m[p_i] / self.ps.density0[p_i]
                    # Keep inverse mass consistent (static solids keep 0 inv mass)
                    if self.ps.is_dynamic[p_i]:
                        self.ps.m_inv[p_i] = 1.0 / (self.ps.m[p_i] + 1e-12)



    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_static_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)

            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                self.compute_boundary_volume_task(p_i, p_j, delta)
            
            self.ps.m_V[p_i] = 1.0 / delta * 3.0

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
        if self.ps.material[p_j] == self.ps.material_solid:
            delta += self.cubic_kernel((self.ps.x[p_i] - self.ps.x[p_j]).norm())


    @ti.kernel
    def compute_moving_boundary_volume(self):
        m = 0.0
        rho0 = 0.0
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                self.compute_boundary_volume_task(p_i, p_j, delta)
            
            self.ps.m_V[p_i] = 1.0 / delta * 3.0



    def initialize(self):


        self.ns.broad_phase()
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()
        self.initialize_boundary_neighbors()

        if self.ps.num_rigid_bodies > 0:
            self.ps.initialize_rigid_mass()

        if hasattr(self.ps, "emitter_system") and self.ps.emitter_system:
            self.ps.emitter_system.reset()



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
        r = x_i - x_j
        h = self.ps.support_radius

        rn = r.norm()
        inv_rn = 1.0 / ti.max(rn, 1e-6 * h)
        r_hat = r * inv_rn

        gradW = self.spiky_kernel_derivative(r)
        vxy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        eps2 = (0.01 * h) * (0.01 * h)

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
                f_vb = k_visc * boundary_viscosity * (self.ps.m[p_j] / self.ps.density[p_i]) * (
                            vxy / (rn * rn + eps2)) * gradW
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
            if self.ps.material[p_i] == self.ps.material_fluid:
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    self.compute_non_pressure_forces_task(p_i, p_j, acc)
            
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
    def update_velocities(self, dt: float):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] = (self.ps.x[p_i] - self.ps.x_old[p_i]) / dt
            else:
                self.ps.v[p_i] = ti.math.vec3(0.0)


    @ti.kernel
    def advect_position(self, dt: float):
        for p_i in ti.grouped(self.ps.x):
            # if self.ps.material[p_i] == self.ps.material_fluid:
            self.ps.x[p_i] = self.ps.x_old[p_i] + dt * self.ps.v[p_i]


    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[p_i] -= (
            1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec



    @ti.kernel
    def enforce_boundary_3D(self, particle_type:int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x[p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x[p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length)


    def forward(self):

        self.ns.broad_phase()
        self.ns.narrow_phase(self.ps.x)

        self.pcg_total_iter = 0
        self.compute_normal()
        self.compute_non_pressure_forces()
        self.advect_velocity(self.dt)

        self.pressure.solve()
        self.advect_position(self.dt)

        self.enforce_boundary_3D(self.ps.material_solid)
        self.enforce_boundary_3D(self.ps.material_fluid)

