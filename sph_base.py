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
        # self.g = np.array(self.ps.cfg.get_cfg("gravitation"))

        self.viscosity = 0.005  # viscosity

        self.density_0 = 1000.0  # reference density
        self.density_0 = self.ps.cfg.get_cfg("density0")
        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4
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

    @ti.kernel
    def precompute_values(self):

        for p_i in ti.grouped(self.ps.x):
            x_i = self.ps.x[p_i]
            for j in range(self.ps.fluid_neighbors_num[p_i]):
                p_j = self.ps.fluid_neighbors[p_i, j]
                x_j = self.ps.x[p_j]
                self.ps.fluid_neighbors_values[p_i, j] = self.nablaWij(x_i - x_j)

    def initialize(self):
        self.ps.initialize_particle_system()
        self.ps.initialize_object_particle_num()
        for r_obj_id in self.ps.object_id_rigid_body:
            self.compute_rigid_rest_cm(r_obj_id)
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()
        self.ps.search_neighbours(self.ps.x)
        # self.ps.initialize_boundary_neighbors(self.density_0)
        # self.ps.initialize_rigid_mass()
        print(f"body mass: {self.ps.body_mass.to_numpy()}")


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
            # self.ps.m_V[p_i] = self.ps.m[p_i] / self.ps.density0[p_i]
            self.ps.m[p_i] = self.ps.m_V[p_i] * self.ps.density0[p_i]
            m = self.ps.m[p_i]
            rho0 = self.ps.density0[p_i]
        
            self.ps.body_mass[self.ps.object_id[p_i]] = m * self.ps.object_particle_num[self.ps.object_id[p_i]]

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
        cm /= sum_m
        return cm
    

    @ti.kernel
    def compute_com_kernel(self, object_id: int)->ti.types.vector(3, float):
        return self.compute_com(object_id)


    # @ti.kernel
    # def solve_constraints(self, object_id: int) -> ti.types.matrix(3, 3, float):
    #     # compute center of mass
    #     cm = self.compute_com(object_id)
    #     # A
    #     A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    #     for p_i in range(self.ps.particle_num[None]):
    #         if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
    #             q = self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id]
    #             p = self.ps.x[p_i] - cm
    #             A += self.ps.m_V0 * self.ps.density[p_i] * p.outer_product(q)

    #     R, S = ti.polar_decompose(A)
        
    #     if all(abs(R) < 1e-6):
    #         R = ti.Matrix.identity(ti.f32, 3)
        
    #     for p_i in range(self.ps.particle_num[None]):
    #         if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
    #         # print("test")
    #             goal = cm + R @ (self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id])
    #             corr = (goal - self.ps.x[p_i]) * 1.0
    #             self.ps.x[p_i] += corr
    #     return R
    
    @ti.kernel
    def solve_constraints(self, dt:float):
        self.ps.R.fill(0.0)
        self.ps.cm.fill(0.0)
        # compute center of mass
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i):
                object_id = self.ps.object_id[p_i]
                self.ps.cm[object_id] += self.ps.m[p_i] * self.ps.x[p_i]

        for object_id in ti.grouped(self.ps.cm):
            self.ps.cm[object_id] /= self.ps.body_mass[object_id]

        for p_i in range(self.ps.particle_num[None]):
            object_id = self.ps.object_id[p_i]
            if self.ps.is_dynamic_rigid_body(p_i):
                q = self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id]
                p = self.ps.x[p_i] - self.ps.cm[object_id]
                self.ps.R[object_id] += self.ps.m_V0 * self.ps.density[p_i] * p.outer_product(q)

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
                corr = (goal - self.ps.x[p_i]) * 1.0
                self.ps.x[p_i] += corr
                # self.ps.v[p_i] += corr / dt
            


    # @ti.kernel
    # def compute_rigid_collision(self):
    #     d = self.ps.particle_diameter
    #     c_r = self.contact_radius

    #     for p_i in range(self.ps.particle_num[None]):
    #         if not self.ps.is_dynamic_rigid_body(p_i):
    #             continue

    #         xcorr = ti.Vector([0.0 for _ in range(self.ps.dim)])
    #         n_sum = ti.Vector([0.0 for _ in range(self.ps.dim)])

    #         for j in range(self.ps.solid_neighbors_num[p_i]):
    #             p_j = self.ps.solid_neighbors[p_i, j]
    #             if self.ps.is_static_rigid_body(p_j):
    #                 r = self.ps.x[p_i] - self.ps.x[p_j]
    #                 dist = r.norm()
    #                 if 1e-9 < dist < c_r:
    #                     n = r / dist
    #                     pen = c_r - dist
    #                     xcorr += pen * n
    #                     n_sum += n

    #                     vn = self.ps.v[p_i].dot(n)
    #                     if vn < 0.0:
    #                         restitution = 0.0
    #                         self.ps.v[p_i] -= (1.0 + restitution) * vn * n

    #         if xcorr.norm() > 0.0:
    #             self.ps.x[p_i] += 0.8 * xcorr

    def solve_rigid_body(self):

        self.solve_constraints()
        # for i in range(1):
        #     # print(self.ps.object_id_rigid_body)
        #     for r_obj_id in self.ps.object_id_rigid_body:
        #         if self.ps.object_collection[r_obj_id]["isDynamic"]:
        #             R = self.solve_constraints(r_obj_id)
        #             # if self.ps.cfg.get_cfg("exportObj"):
        #             #     # For output obj only: update the mesh
        #             #     cm = self.compute_com_kernel(r_obj_id)
        #             #     ret = R.to_numpy() @ (self.ps.object_collection[r_obj_id]["restPosition"] - self.ps.object_collection[r_obj_id]["restCenterOfMass"]).T
        #             #     self.ps.object_collection[r_obj_id]["mesh"].vertices = cm.to_numpy() + ret.T
        # self.compute_rigid_collision()




    def step(self):
        self.ps.initialize_particle_system()
        self.compute_moving_boundary_volume()
        self.substep()

        self.enforce_boundary_3D(self.ps.material_solid)
        if self.ps.dim == 2:
            self.enforce_boundary_2D(self.ps.material_fluid)
        # elif self.ps.dim == 3:
        #     self.enforce_boundary_3D(self.ps.material_fluid)
