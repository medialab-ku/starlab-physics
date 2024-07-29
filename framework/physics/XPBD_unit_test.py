import csv
import taichi as ti
import numpy as np
from ..physics import collision_constraints_x, collision_constraints_v, col_test
from ..collision.lbvh_cell import LBVH_CELL

@ti.data_oriented
class Solver:
    def __init__(self,
                 dHat,
                 stiffness_stretch,
                 stiffness_bending,
                 g,
                 dt):

        self.g = g
        self.dt = dt
        self.dHat = dHat
        self.stiffness_stretch = stiffness_stretch
        self.stiffness_bending = stiffness_bending
        self.damping = 0.001
        self.mu = 0.1
        self.padding = 0.05

        self.enable_velocity_update = False
        self.enable_collision_handling = True

        self.x_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.x_test_st = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.v_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.dx_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.nc_test = ti.field(dtype=ti.f32, shape=3)
        self.color_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.y_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)

        self.edge_indices_test = ti.Vector.field(n=2, dtype=ti.i32, shape=3)
        self.face_indices_test = ti.field(dtype=ti.i32, shape=3)
        self.l0_test = ti.field(dtype=ti.f32, shape=3)
        self.init_test()

    @ti.kernel
    def init_test(self):
        size = 2.0
        self.x_test[0] = ti.math.vec3([0.0, 0.0, 0.0])
        self.x_test[1] = ti.math.vec3([size, 0.0, 0.0])
        self.x_test[2] = ti.math.vec3([0.5 * size, 0.0, 0.5 * ti.sqrt(3.0) * size])

        center = ti.math.vec3([0.0, 0.0, 0.0])
        for i in range(3):
            center += self.x_test[i]

        center /= 3.0

        for i in range(3):
            self.x_test[i] -= center


        translate = ti.math.vec3([0.0, -2.0, 0.0])
        self.x_test_st[0] = ti.math.vec3([0.0, 0.0, 0.0])
        self.x_test_st[1] = ti.math.vec3([size, 0.0, 0.0])
        self.x_test_st[2] = ti.math.vec3([0.5 * size, 0.0, 0.5 * ti.sqrt(3.0) * size])

        rotate = ti.math.vec3([0.0, 90.0, 0.0])
        for i in range(3):
            xi = self.x_test[i]
            v_4d = ti.math.vec4([xi, 1.0])
            rot_rad_x = ti.math.radians(rotate[0])
            rot_rad_y = ti.math.radians(rotate[1])
            rot_rad_z = ti.math.radians(rotate[2])
            rotated_x = ti.math.rotation3d(rot_rad_x, rot_rad_y, rot_rad_z) @ v_4d
            self.x_test[i] = ti.math.vec3([rotated_x[0], rotated_x[1], rotated_x[2]])

        center = ti.math.vec3([0.0, 0.0, 0.0])
        for i in range(3):
            center += self.x_test_st[i]

        center /= 3.0

        for i in range(3):
            self.x_test_st[i] -= center

        for i in range(3):
            self.x_test_st[i] += translate


        self.edge_indices_test[0][0] = 0
        self.edge_indices_test[0][1] = 1
        self.edge_indices_test[1][0] = 1
        self.edge_indices_test[1][1] = 2
        self.edge_indices_test[2][0] = 2
        self.edge_indices_test[2][1] = 0

        self.l0_test.fill(size)

        self.color_test[0] = ti.math.vec3([1.0, 0.0, 0.0])
        self.color_test[1] = ti.math.vec3([0.0, 1.0, 0.0])
        self.color_test[2] = ti.math.vec3([0.0, 0.0, 1.0])

        self.v_test.fill(0.0)

        self.face_indices_test[3 * 0 + 0] = 0
        self.face_indices_test[3 * 0 + 1] = 1
        self.face_indices_test[3 * 0 + 2] = 2



    def reset(self):
        self.init_test()


    @ti.kernel
    def compute_y_test(self, dt: ti.f32):
        for i in range(3):
            self.y_test[i] = self.x_test[i] + self.v_test[i] * dt + self.g * dt * dt

    def solve_constraints_test(self):

        self.nc_test.fill(0)
        self.dx_test.fill(0.)
        self.solve_stretch_test()

        self.update_dx_test()

        self.nc_test.fill(0)
        self.dx_test.fill(0.)

        self.solve_collision_test()

        self.update_dx_test()

    @ti.kernel
    def solve_stretch_test(self):

        compliance_stretch = 1e7
        for i in range(3):
            bi = i
            l0 = self.l0_test[bi]
            v0, v1 = self.edge_indices_test[bi][0], self.edge_indices_test[bi][1]
            x10 = self.y_test[v0] - self.y_test[v1]
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            schur = (1.0 + 1.0) * nabla_C.dot(nabla_C)

            ld = compliance_stretch * C / (compliance_stretch * schur + 1.0)

            self.dx_test[v0] -= ld * nabla_C
            self.dx_test[v1] += ld * nabla_C
            self.nc_test[v0] += 1.0
            self.nc_test[v1] += 1.0

    @ti.kernel
    def solve_collision_test(self):

        compliance_collision = 1e7
        for i in range(3):
            for j in range(3):
                col_test.__ee_st(compliance_collision, i, j, self.edge_indices_test, self.edge_indices_test, self.y_test, self.x_test_st, self.dx_test, self.nc_test, self.dHat)


    @ti.func
    def is_in_face(self, vid, fid):

        v1 = self.mesh_dy.face_indices[3 * fid + 0]
        v2 = self.mesh_dy.face_indices[3 * fid + 1]
        v3 = self.mesh_dy.face_indices[3 * fid + 2]

        return (v1 == vid) or (v2 == vid) or (v3 == vid)

    @ti.func
    def share_vertex(self, ei0, ei1):

        v0 = self.mesh_dy.edge_indices[2 * ei0 + 0]
        v1 = self.mesh_dy.edge_indices[2 * ei0 + 1]
        v2 = self.mesh_dy.edge_indices[2 * ei1 + 0]
        v3 = self.mesh_dy.edge_indices[2 * ei1 + 1]

        return (v0 == v2) or (v0 == v3) or (v1 == v2) or (v1 == v3)


    @ti.kernel
    def solve_spring_constraints_x(self, compliance_stretch: ti.f32, compliance: ti.f32):

        for i in range(self.max_num_edges_dy + self.mesh_dy.bending_constraint_count):

            # solve stretch constraints
            if i < self.max_num_edges_dy:
                bi = i
                l0 = self.mesh_dy.edges.l0[bi]
                v0, v1 = self.mesh_dy.edge_indices[2 * bi], self.mesh_dy.edge_indices[2 * bi + 1]
                x10 = self.mesh_dy.verts.y[v0] - self.mesh_dy.verts.y[v1]
                lij = x10.norm()

                C = (lij - l0)
                nabla_C = x10.normalized()
                schur = (self.mesh_dy.verts.fixed[v0] * self.mesh_dy.verts.m_inv[v0] + self.mesh_dy.verts.fixed[v1] * self.mesh_dy.verts.m_inv[v1]) * nabla_C.dot(nabla_C)

                ld = compliance_stretch * C / (compliance_stretch * schur + 1.0)

                self.mesh_dy.verts.dx[v0] -= self.mesh_dy.verts.fixed[v0] * self.mesh_dy.verts.m_inv[v0] * ld * nabla_C
                self.mesh_dy.verts.dx[v1] += self.mesh_dy.verts.fixed[v1] * self.mesh_dy.verts.m_inv[v1] * ld * nabla_C
                self.mesh_dy.verts.nc[v0] += 1.0
                self.mesh_dy.verts.nc[v1] += 1.0

            # solve stretch constraints
            else:
                bi = i - self.max_num_edges_dy
                v0, v1 = self.mesh_dy.bending_indices[2 * bi], self.mesh_dy.bending_indices[2 * bi + 1]
                l0 = self.mesh_dy.bending_l0[bi]
                x10 = self.mesh_dy.verts.x[v0] - self.mesh_dy.verts.x[v1]
                lij = x10.norm()

                C = (lij - l0)
                nabla_C = x10.normalized()

                e_v0_fixed, e_v1_fixed = self.mesh_dy.verts.fixed[v0], self.mesh_dy.verts.fixed[v1]
                e_v0_m_inv, e_v1_m_inv = self.mesh_dy.verts.m_inv[v0], self.mesh_dy.verts.m_inv[v1]

                schur = (e_v0_fixed * e_v0_m_inv + e_v1_fixed * e_v1_m_inv) * nabla_C.dot(nabla_C)
                ld = compliance * C / (compliance * schur + 1.0)

                self.mesh_dy.verts.dx[v0] -= e_v0_fixed * e_v0_m_inv * ld * nabla_C
                self.mesh_dy.verts.dx[v1] += e_v1_fixed * e_v1_m_inv * ld * nabla_C
                self.mesh_dy.verts.nc[v0] += 1.0
                self.mesh_dy.verts.nc[v1] += 1.0



    @ti.kernel
    def compute_velocity_test(self, damping: ti.f32, dt: ti.f32):
        for i in range(3):
            self.v_test[i] = (1.0 - damping) * (self.y_test[i] - self.x_test[i]) / dt
            self.x_test[i] = self.y_test[i]


    @ti.kernel
    def update_dx_test(self):
        for i in range(3):
            if self.nc_test[i] > 0:
                self.y_test[i] += (self.dx_test[i] / self.nc_test[i])



    def forward(self, n_substeps):

        dt_sub = self.dt / n_substeps

        for _ in range(n_substeps):
            self.compute_y_test(dt_sub)

            self.solve_constraints_test()

            self.compute_velocity_test(damping=self.damping, dt=dt_sub)
