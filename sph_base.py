from matplotlib.pyplot import axis
import taichi as ti
import numpy as np


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.ps.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        elif self.ps.dim == 3:
            self.g = ti.Vector([0.0, -9.81, 0.0])
        try:
            g_cfg = self.ps.cfg.get_cfg("gravitation")
            if isinstance(g_cfg, (list, tuple, np.ndarray)):
                if self.ps.dim == 2 and len(g_cfg) >= 2:
                    self.g = ti.Vector([float(g_cfg[0]), float(g_cfg[1])])
                elif self.ps.dim == 3 and len(g_cfg) >= 3:
                    self.g = ti.Vector([float(g_cfg[0]), float(g_cfg[1]), float(g_cfg[2])])
        except Exception:
            pass

        self.viscosity = 0.01  # viscosity
        self.surface_tension = 0.005
        self.adhesion_coeff = self.surface_tension

        self.density_0 = 1000.0  # reference density
        self.density_0 = self.ps.cfg.get_cfg("density0")
        self.time = 0.0
        self.dt = self.ps.cfg.get_cfg("timeStepSize")
        self.nablaWij = self.spiky_kernel_derivative



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
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    @ti.func
    def cohesion_term(self, r):
        h = self.ps.support_radius
        rn = r.norm()
        res = 0.0
        k = 32.0 / (np.pi * h**9)
        if rn <= 0.0 or rn > h:
            res = 0.0
            
        elif rn > 0.5 * h:
            t = (h - rn) * rn
            res = k * (t * t * t)
        else:
            t = 2.0 * (h - rn) * rn
            res = k * ((t * t * t) - (h**6) / 64.0)

        return res

    @ti.func
    def adhesion_term(self, r):
        h = self.ps.support_radius
        rn = r.norm()
        k = 0.007 / (h**3.25)
        arg = -4.0 * rn * rn / h + 6.0 * rn - 2.0 * h
        arg = ti.max(arg, 0.0)
        res = 0.0
        if rn <= h and rn > 0.5 * h:
            res = k * ti.sqrt(ti.sqrt(arg))

        return res


    def initialize(self):
        self.ps.initialize_particle_system()
        self.ps.initialize_object_particle_num()

        if self.ps.num_rigid_bodies > 0:        
            for r_obj_id in self.ps.object_id_rigid_body:
                self.compute_rigid_rest_cm(r_obj_id)

        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()
        self.ps.initialize_boundary_neighbors()

        if self.ps.num_rigid_bodies > 0:
            self.ps.initialize_rigid_mass()

        if hasattr(self.ps, "emitter_system") and self.ps.emitter_system:
            self.ps.emitter_system.reset()



    @ti.kernel
    def compute_rigid_rest_cm(self, object_id: int):
        self.ps.rigid_rest_cm[object_id] = self.compute_com(object_id)

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_static_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

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
            self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly


    def substep(self):
        pass


    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[p_i] -= (
            1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec


    @ti.kernel
    def enforce_boundary_2D(self, particle_type:int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]: 
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
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
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(
                            p_i, collision_normal / collision_normal_length)

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


    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V[p_i] * self.ps.density0[p_i]
                cm += mass * self.ps.x[p_i]
                sum_m += mass

        if sum_m > 1e-12:
            cm /= sum_m
        return cm
    

    @ti.kernel
    def compute_com_kernel(self, object_id: int)->ti.types.vector(3, float):
        return self.compute_com(object_id)


    @ti.kernel
    def solve_constraints(self):
        self.ps.R.fill(0.0)
        self.ps.cm.fill(0.0)
        # compute center of mass
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i):
                object_id = self.ps.object_id[p_i]
                self.ps.cm[object_id] += self.ps.m[p_i] * self.ps.x[p_i]

        for object_id in ti.grouped(self.ps.cm):
            self.ps.cm[object_id] /= self.ps.mass_rb[object_id]

        for p_i in range(self.ps.particle_num[None]):
            object_id = self.ps.object_id[p_i]
            if self.ps.is_dynamic_rigid_body(p_i):
                q = self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id]
                p = self.ps.x[p_i] - self.ps.cm[object_id]
                self.ps.R[object_id] += self.ps.m[p_i] * p.outer_product(q)

        for object_id in ti.grouped(self.ps.R):
                A = self.ps.R[object_id]
                R, S = ti.polar_decompose(A)
                if all(abs(R) < 1e-6):
                    R = ti.Matrix.identity(ti.f32, 3)
                self.ps.R[object_id] = R

        for p_i in range(self.ps.particle_num[None]):
            object_id = self.ps.object_id[p_i]
            if self.ps.is_dynamic_rigid_body(p_i):
                goal = self.ps.cm[object_id] + self.ps.R[object_id] @ (self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id])
                corr = (goal - self.ps.x[p_i])

                self.ps.x[p_i] += corr


    @ti.kernel
    def apply_rigid_pressure(self, dt: float):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            if self.ps.density[p_i] <= self.ps.density0[p_i]:
                continue
            p_i_val = self.p[p_i]
            density_i_sq = self.ps.density[p_i] * self.ps.density[p_i]
            if p_i_val <= 0.0:
                continue
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                f_b = ti.math.vec3(0.0)
                p_j = self.ps.fluid_neighbors[p_i, j]
                body = self.ps.object_id[p_j]
                if self.ps.is_dynamic_rigid_body(p_j):
                    grad = self.ps.fluid_neighbors_values[p_i, j]
                    n = grad.normalized()
                    m_ij = self.ps.m[p_i] * self.ps.m[p_j]

                    f_b = (m_ij * p_i_val / density_i_sq) * grad
                    dv = dt * f_b / (self.ps.m[p_j] + 1e-12)

                    self.ps.v[p_j] += dv


    @ti.kernel
    def rigid_compute_cm_and_vcm(self):
        # com, v_cm
        self.ps.cm.fill(0.0)
        self.vsum_rb.fill(0.0)
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic_rigid_body(p_i):
                obj = self.ps.object_id[p_i]
                mi = self.ps.m[p_i]
                ti.atomic_add(self.ps.cm[obj], mi * self.ps.x[p_i])
                ti.atomic_add(self.vsum_rb[obj], mi * self.ps.v[p_i])
        for obj in ti.grouped(self.ps.cm):
            M = self.ps.mass_rb[obj]
            invM = 1.0 / (M + 1e-12)
            self.ps.cm[obj] *= invM
            self.ps.v_cm_rb[obj] = self.vsum_rb[obj] * invM

    @ti.kernel
    def rigid_compute_angular_velocity(self):
        # omega = I^-1 t
        self.I_rb.fill(0.0)
        self.t_rb.fill(0.0)
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic_rigid_body(p_i):
                obj = self.ps.object_id[p_i]
                mi = self.ps.m[p_i]
                r  = self.ps.x[p_i] - self.ps.cm[obj]
                v_rel = self.ps.v[p_i] - self.ps.v_cm_rb[obj]
                rrT = r.outer_product(r)
                I3 = ti.math.mat3([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
                self.I_rb[obj] += mi * ((r.dot(r)) * I3 - rrT)
                self.t_rb[obj] += mi * ti.math.cross(r, v_rel)
        for obj in ti.grouped(self.ps.cm):
            I = self.I_rb[obj]
            eps = 1e-8
            I += ti.math.mat3([[eps,0.0,0.0],[0.0,eps,0.0],[0.0,0.0,eps]])
            self.ps.omega_rb[obj] = I.inverse() @ self.t_rb[obj]

    @ti.kernel
    def rigid_project_velocities(self):
        # v = v_cm + omega × r
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic_rigid_body(p_i):
                obj = self.ps.object_id[p_i]
                r = self.ps.x[p_i] - self.ps.cm[obj]
                self.ps.v[p_i] = self.ps.v_cm_rb[obj] + ti.math.cross(self.ps.omega_rb[obj], r)

    def step(self):

        if hasattr(self.ps, 'emitter_system') and self.ps.emitter_system is not None:
            dt = self.dt
            self.ps.emitter_system.step(self.time, dt)
            self.time += dt

        self.ps.initialize_particle_system()
        self.compute_moving_boundary_volume()
        self.substep()
        self.enforce_boundary_3D(self.ps.material_solid)

        if self.ps.dim == 2:
            self.enforce_boundary_2D(self.ps.material_fluid)
        elif self.ps.dim == 3:
            self.enforce_boundary_3D(self.ps.material_fluid)
