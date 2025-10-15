import time
import numpy as np
import taichi as ti
import sph_kernel

@ti.data_oriented
class Viscosity:
    def __init__(self, particle_system):
        # super().__init__(particle_system)

        self.ps = particle_system
        self.gradW = sph_kernel.spiky_kernel_derivative

    @ti.kernel
    def solve(self, coeff: float, dt: float):

        h = self.ps.support_radius
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue

            acc = ti.math.vec3(0.0)
            if self.ps.material[p_i] == self.ps.material_fluid:

                x_i = self.ps.x[p_i]
                for j in range(self.ps.particle_neighbors_num[p_i]):
                    p_j = self.ps.particle_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]
                    r = x_i - x_j

                    rn = r.norm()
                    vxy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
                    eps2 = (0.01 * h) * (0.01 * h)

                    # ---------- Cohesion & Viscosity ----------
                    # Akinci2012
                    # Akinci2013
                    if self.ps.material[p_j] == self.ps.material_fluid:

                        # Neighborhood deficiency correction K_ij
                        K_ij = 2.0 * self.ps.density0[p_i] / (self.ps.density[p_i] + self.ps.density[p_j])

                        # Viscosity (fluid-fluid)
                        k_visc = 2.0 * (self.ps.dim + 2.0)
                        f_v = k_visc * coeff * (self.ps.m[p_j] / self.ps.density[p_j]) * (vxy / (rn * rn + eps2)) * self.gradW(r, h)
                        acc += K_ij * f_v

                self.ps.v[p_i] += acc * dt
