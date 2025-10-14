import numpy as np
import taichi as ti
import sph_kernel

@ti.data_oriented
class SurfaceTension:
    def __init__(self, particle_system):

        self.ps = particle_system
        self.gradW = sph_kernel.spiky_kernel_derivative
        # self.W   = sph_kernel.cubic_kernel

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
    def solve(self, coeff_sf: float, coeff_adh: float, dt: float):

        #compute normal
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                self.ps.n[p_i] = ti.Vector.zero(float, self.ps.dim)
                continue
            n = ti.Vector.zero(float, self.ps.dim)
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                if self.ps.material[p_j] != self.ps.material_fluid:
                    continue
                n += (self.ps.m[p_j] / (self.ps.density[p_j] + 1e-12)) * self.gradW(self.ps.x[p_i] - self.ps.x[p_j], self.ps.support_radius)
            self.ps.n[p_i] = self.ps.support_radius * n

        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                # self.ps.acceleration[p_i].fill(0.0)
                continue

            acc = ti.math.vec3(0.0)

            if self.ps.material[p_i] == self.ps.material_fluid:

                x_i = self.ps.x[p_i]
                for j in range(self.ps.fluid_neighbors_num[p_i]):
                    p_j = self.ps.fluid_neighbors[p_i, j]
                    x_j = self.ps.x[p_j]
                    r = x_i - x_j
                    h = self.ps.support_radius

                    rn = r.norm()
                    inv_rn = 1.0 / ti.max(rn, 1e-6 * h)
                    r_hat = r * inv_rn

                    # gradW = self.spiky_kernel_derivative(r)
                    vxy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
                    eps2 = (0.01 * h) * (0.01 * h)

                    # ---------- Cohesion & Viscosity ----------
                    # Akinci2012
                    # Akinci2013
                    if self.ps.material[p_j] == self.ps.material_fluid:
                        # Cohesion
                        f_coh = - coeff_sf * self.ps.m[p_i] * self.ps.m[p_j] * self.cohesion_term(r) * r_hat
                        # Curvature
                        f_curv = -coeff_sf * self.ps.m[p_i] * (self.ps.n[p_i] - self.ps.n[p_j])
                        # Neighborhood deficiency correction K_ij
                        K_ij = 2.0 * self.ps.density0[p_i] / (self.ps.density[p_i] + self.ps.density[p_j])
                        # K_ij = ti.min(K_ij, 1.0)
                        acc += K_ij * (f_coh + f_curv)


                    # ---------- Adhesion ----------
                    elif self.ps.material[p_j] == self.ps.material_solid:
                        f_adh = - coeff_adh * self.ps.m[p_i] * self.ps.m[p_j] * self.adhesion_term(r) * r_hat
                        acc += f_adh

                        # (Optional) Boundary Viscosity
                        boundary_viscosity = 0.0
                        if boundary_viscosity > 0.0:
                            k_visc = 2.0 * (self.ps.dim + 2.0)
                            f_vb = k_visc * boundary_viscosity * (self.ps.m[p_j] / self.ps.density[p_i]) * (vxy / (rn * rn + eps2)) * self.gradW(r, self.ps.support_radius)
                            acc += f_vb

                self.ps.v[p_i] += acc * dt