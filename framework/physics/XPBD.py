import csv
import taichi as ti
import numpy as np
from ..physics import collision_constraints_x, collision_constraints_v
from ..collision.lbvh_cell import LBVH_CELL

@ti.data_oriented
class Solver:
    def __init__(self,
                 mesh_dy,
                 mesh_st,
                 dHat,
                 stiffness_stretch,
                 stiffness_bending,
                 g,
                 dt):

        self.mesh_dy = mesh_dy
        self.mesh_st = mesh_st
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
        self.enable_move_obstacle = False
        self.export_mesh = False


        self.max_num_verts_dy = len(self.mesh_dy.verts)
        self.max_num_edges_dy = len(self.mesh_dy.edges)
        self.max_num_faces_dy = len(self.mesh_dy.faces)

        self.max_num_verts_st = 0
        self.max_num_edges_st = 0
        self.max_num_faces_st = 0


        self.x_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.x_test_st = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.v_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.color_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.y_test = ti.Vector.field(n=3, dtype=ti.f32, shape=3)

        self.edge_indices_test = ti.Vector.field(n=2, dtype=ti.i32, shape=3)
        self.face_indices_test = ti.field(dtype=ti.i32, shape=3)
        self.l0_test = ti.field(dtype=ti.f32, shape=3)


        self.lbvh_st = None
        if self.mesh_st != None:
            self.lbvh_st = LBVH_CELL(len(self.mesh_st.faces))
            self.max_num_verts_st = len(self.mesh_st.verts)
            self.max_num_edges_st = len(self.mesh_st.edges)
            self.max_num_faces_st = len(self.mesh_st.faces)

        self.lbvh_dy = LBVH_CELL(len(self.mesh_dy.faces))

        self.vt_st_pair_cache_size = 40
        self.vt_st_candidates = ti.field(dtype=ti.int32, shape=(self.max_num_verts_dy, self.vt_st_pair_cache_size))
        self.vt_st_candidates_num = ti.field(dtype=ti.int32, shape=self.max_num_verts_dy)
        self.vt_st_pair = ti.field(dtype=ti.int32, shape=(self.max_num_verts_dy, self.vt_st_pair_cache_size, 2))
        self.vt_st_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_verts_dy)
        self.vt_st_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_verts_dy, self.vt_st_pair_cache_size, 4))
        self.vt_st_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_verts_dy, self.vt_st_pair_cache_size))

        self.tv_st_pair_cache_size = 40
        self.tv_st_candidates = ti.field(dtype=ti.int32, shape=(self.max_num_verts_st, self.vt_st_pair_cache_size))
        self.tv_st_candidates_num = ti.field(dtype=ti.int32, shape=self.max_num_verts_st)
        self.tv_st_pair = ti.field(dtype=ti.int32, shape=(self.max_num_verts_st, self.tv_st_pair_cache_size, 2))
        self.tv_st_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_verts_st)
        self.tv_st_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_verts_st, self.tv_st_pair_cache_size, 4))
        self.tv_st_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_verts_st, self.tv_st_pair_cache_size))

        self.vt_dy_pair_cache_size = 40
        self.vt_dy_candidates = ti.field(dtype=ti.int32, shape=(self.max_num_verts_dy, self.vt_dy_pair_cache_size))
        self.vt_dy_candidates_num = ti.field(dtype=ti.int32, shape=self.max_num_verts_dy)
        self.vt_dy_pair = ti.field(dtype=ti.int32, shape=(self.max_num_verts_dy, self.vt_st_pair_cache_size, 2))
        self.vt_dy_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_verts_dy)
        self.vt_dy_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_verts_dy, self.vt_st_pair_cache_size, 4))
        self.vt_dy_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_verts_dy, self.vt_st_pair_cache_size))


        self.ee_static_pair_cache_size = 40
        self.ee_st_candidates = ti.field(dtype=ti.int32, shape=(self.max_num_edges_dy, self.ee_static_pair_cache_size))
        self.ee_st_candidates_num = ti.field(dtype=ti.int32, shape=self.ee_static_pair_cache_size)
        self.ee_static_pair = ti.field(dtype=ti.int32, shape=(self.max_num_edges_dy, self.ee_static_pair_cache_size, 2))
        self.ee_static_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_edges_dy)
        self.ee_static_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_edges_dy, self.ee_static_pair_cache_size, 4))
        self.ee_static_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_edges_dy, self.ee_static_pair_cache_size))
        #
        # self.tv_dynamic_pair_cache_size = 40
        # self.tv_dynamic_pair = ti.field(dtype=ti.int32, shape=(self.max_num_faces_dynamic, self.tv_dynamic_pair_cache_size, 2))
        # self.tv_dynamic_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_faces_dynamic)
        # self.tv_dynamic_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_faces_dynamic, self.tv_dynamic_pair_cache_size, 4))
        # self.tv_dynamic_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_faces_dynamic, self.tv_dynamic_pair_cache_size))
        #
        # self.ee_dynamic_pair_cache_size = 40
        # self.ee_dynamic_pair = ti.field(dtype=ti.int32, shape=(self.max_num_edges_dynamic, self.ee_dynamic_pair_cache_size, 2))
        # self.ee_dynamic_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_edges_dynamic)
        # self.ee_dynamic_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.ee_dynamic_pair_cache_size, 4))
        # self.ee_dynamic_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.ee_dynamic_pair_cache_size))

        # self.frame = ti.field(dtype=ti.i32, shape=1)
        # self.frame[0] = 0
        #
        # self.max_num_anim = 40
        # self.num_animation = ti.field(dtype=ti.i32, shape=4)   # maximum number of handle set
        # self.cur_animation = ti.field(dtype=ti.i32, shape=4)   # maximum number of handle set
        # self.anim_rotation_mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=4)
        #
        # self.anim_local_origin = ti.Vector.field(3, dtype=ti.f32, shape=4)
        # self.active_anim_frame = ti.field(dtype=ti.i32, shape=(4, self.max_num_anim)) # maximum number of animation
        # self.action_anim = ti.Vector.field(6, dtype=ti.f32, shape=(4, self.max_num_anim)) # maximum number of animation, a animation consist (vx,vy,vz,rx,ry,rz)
        # self.anim_x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)

        self.sewing_pairs = ti.Vector.field(n=2, dtype=ti.int32, shape=(self.max_num_verts_dy))
        self.sewing_pairs_num = ti.field(dtype=ti.int32, shape=())
        self.sewing_pairs_num[None] = 0

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

        rotate = ti.math.vec3([0.0, 0.0, 60.0])
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
        self.mesh_dy.reset()
        if self.mesh_st != None:
            self.mesh_st.reset()


    @ti.kernel
    def compute_y(self, dt: ti.f32):
        for v in self.mesh_dy.verts:
            v.y = v.x + v.fixed * v.v * dt + self.g * dt * dt

    @ti.kernel
    def compute_y_test(self, dt: ti.f32):
        for i in range(3):
            self.y_test[i] = self.x_test[i] + self.v_test[i]  * dt + self.g * dt * dt

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
    def broadphase_lbvh(self, padding: ti.f32, cell_size_st: ti.math.vec3, origin_st: ti.math.vec3, cell_size_dy: ti.math.vec3, origin_dy: ti.math.vec3):

        self.vt_st_candidates_num.fill(0)
        self.tv_st_candidates_num.fill(0)
        self.vt_dy_candidates_num.fill(0)

        for i in range(2 * self.max_num_verts_dy + self.max_num_verts_st):

            if i < self.max_num_verts_dy:
                vid = i
                y = self.mesh_dy.verts.y[vid]
                aabb_min = y - padding * ti.math.vec3(1.0)
                aabb_max = y + padding * ti.math.vec3(1.0)
                self.lbvh_st.traverse_cell_bvh_single(cell_size_st, origin_st, aabb_min, aabb_max, vid, self.vt_st_candidates, self.vt_st_candidates_num)

            elif i < 2 * self.max_num_verts_dy:
                vid = i - self.max_num_verts_dy
                y = self.mesh_dy.verts.y[vid]
                aabb_min = y - padding * ti.math.vec3(1.0)
                aabb_max = y + padding * ti.math.vec3(1.0)
                self.lbvh_dy.traverse_cell_bvh_single(cell_size_dy, origin_dy, aabb_min, aabb_max, vid, self.vt_dy_candidates, self.vt_dy_candidates_num)

            else:
                vid = i - 2 * self.max_num_verts_dy
                x = self.mesh_st.verts.x[vid]
                aabb_min = x - padding * ti.math.vec3(1.0)
                aabb_max = x + padding * ti.math.vec3(1.0)
                self.lbvh_dy.traverse_cell_bvh_single(cell_size_dy, origin_dy, aabb_min, aabb_max, vid, self.tv_st_candidates, self.tv_st_candidates_num)

        self.ee_st_candidates_num.fill(0)
        for i in range(self.max_num_edges_dy):
            v0, v1 = self.mesh_dy.edge_indices[2 * i], self.mesh_dy.edge_indices[2 * i + 1]
            aabb_min = ti.math.min(self.mesh_dy.verts.y[v0], self.mesh_dy.verts.y[v1])
            aabb_max = ti.math.max(self.mesh_dy.verts.y[v0], self.mesh_dy.verts.y[v1])
            self.lbvh_st.traverse_cell_bvh_single(cell_size_st, origin_st, aabb_min, aabb_max, i, self.ee_st_candidates, self.ee_st_candidates_num)



    @ti.kernel
    def solve_collision_constraints_x(self, compliance_col: ti.f32):

        self.vt_st_pair_num.fill(0)
        self.tv_st_pair_num.fill(0)
        self.vt_dy_pair_num.fill(0)

        d = self.dHat
        for i in range(2 * self.max_num_verts_dy + self.max_num_verts_st):

            if i < self.max_num_verts_dy:
                vid = i
                for j in range(self.vt_st_candidates_num[vid]):
                    fi_s = self.vt_st_candidates[vid, j]
                    collision_constraints_x.__vt_st(compliance_col, vid, fi_s, self.mesh_dy, self.mesh_st, d, self.vt_st_pair_cache_size, self.vt_st_pair, self.vt_st_pair_num, self.vt_st_pair_g, self.vt_st_pair_schur)

            elif i < 2 * self.max_num_verts_dy:
                vid = i - self.max_num_verts_dy
                # for j in range(self.vt_dy_candidates_num[vid]):
                #     fi_d = self.vt_dy_candidates[vid, j]
                #     if self.is_in_face(vid, fi_d) != True:
                #         collision_constraints_x.__vt_dy(compliance_col, vid, fi_d, self.mesh_dy, d, self.vt_st_pair_cache_size, self.vt_dy_pair, self.vt_dy_pair_num, self.vt_dy_pair_g, self.vt_dy_pair_schur)
            else:
                vis = i - 2 * self.max_num_verts_dy
                # for j in range(self.tv_st_candidates_num[vis]):
                #     fi_d = self.tv_st_candidates[vis, j]
                #     collision_constraints_x.__tv_st(compliance_col, fi_d, vis, self.mesh_dy, self.mesh_st, d, self.vt_st_pair_cache_size, self.tv_st_pair, self.tv_st_pair_num, self.tv_st_pair_g, self.tv_st_pair_schur)

        for eid in range(self.max_num_edges_dy):
            # for j in range(self.ee_st_candidates_num[eid]):
            #     fis = self.ee_st_candidates[eid, j]
            #     for k in range(3):
            for eis in range(self.max_num_edges_st):
                    # eis = self.mesh_st.face_edge_indices[3 * fis + k]
                    collision_constraints_x.__ee_st(compliance_col, eid, eis, self.mesh_dy, self.mesh_st, d)


    @ti.kernel
    def solve_collision_constraints_v(self, mu: ti.f32):

        for i in range(2 * self.max_num_verts_dy + self.max_num_verts_st):

            if i < self.max_num_verts_dy:
                vid = i
        # for vid in range(self.max_num_verts_dy):
                for j in range(self.vt_st_pair_num[vid]):
                    fi_s, dtype = self.vt_st_pair[vid, j, 0], self.vt_st_pair[vid, j, 1]
                    g0, g1, g2, g3 = self.vt_st_pair_g[vid, j, 0], self.vt_st_pair_g[vid, j, 1], self.vt_st_pair_g[vid, j, 2], self.vt_st_pair_g[vid, j, 3]
                    schur = self.vt_st_pair_schur[vid, j]
                    collision_constraints_v.__vt_st(vid, fi_s, dtype, self.mesh_dy, self.mesh_st, g0, g1, g2, g3, schur, mu)

            elif i < 2 * self.max_num_verts_dy:
                vid = i - self.max_num_verts_dy
                for j in range(self.vt_dy_pair_num[vid]):
                    fi_d, dtype = self.vt_dy_pair[vid, j, 0], self.vt_dy_pair[vid, j, 1]
                    g0, g1, g2, g3 = self.vt_dy_pair_g[vid, j, 0], self.vt_dy_pair_g[vid, j, 1], self.vt_dy_pair_g[vid, j, 2], self.vt_dy_pair_g[vid, j, 3]
                    schur = self.vt_dy_pair_schur[vid, j]
                    collision_constraints_v.__vt_dy(vid, fi_d, dtype, self.mesh_dy, g0, g1, g2, g3, schur, mu)

            else:
                vis = i - 2 * self.max_num_verts_dy
                for j in range(self.tv_st_pair_num[vis]):
                    fi_d, dtype = self.tv_st_pair[vis, j, 0], self.tv_st_pair[vis, j, 1]
                    g0, g1, g2, g3 = self.tv_st_pair_g[vis, j, 0],  self.tv_st_pair_g[vis, j, 1],  self.tv_st_pair_g[vis, j, 2],  self.tv_st_pair_g[vis, j, 3]
                    schur = self.tv_st_pair_schur[vis, j]
                    collision_constraints_v.__tv_st(fi_d, vis, dtype, self.mesh_dy, self.mesh_st, g0, g1, g2, g3, schur, mu)

    @ti.kernel
    def update_x(self, dt: ti.f32):

        for v in self.mesh_dy.verts:
            # if v.id != 0:
            v.x += dt * v.v


    @ti.kernel
    def compute_velocity(self, damping: ti.f32, dt: ti.f32):
        for v in self.mesh_dy.verts:
            v.v = (1.0 - damping) * v.fixed * (v.y - v.x) / dt

    @ti.kernel
    def compute_velocity_test(self, damping: ti.f32, dt: ti.f32):
        for i in range(3):
            self.v_test[i] = (1.0 - damping) * (self.y_test[i] - self.x_test[i]) / dt
            self.x_test[i] = self.y_test[i]


    def init_variables(self):

        self.mesh_dy.verts.dx.fill(0.0)
        self.mesh_dy.verts.nc.fill(0.0)

    def solve_constraints_jacobi_x(self, dt):

        self.init_variables()

        compliance_stretch = self.stiffness_bending * dt * dt
        compliance_bending = self.stiffness_stretch * dt * dt

        self.solve_spring_constraints_x(compliance_stretch, compliance_bending)

        self.update_dx()

        self.init_variables()
        if self.enable_collision_handling:

            self.broadphase_lbvh(self.padding, self.lbvh_st.cell_size, self.lbvh_st.origin, self.lbvh_dy.cell_size, self.lbvh_dy.origin)
            compliance_collision = 1e8
            self.solve_collision_constraints_x(compliance_collision)

        # self.update_dx()

        # compliance_sewing = 0.5 * self.YM * dt * dt
        # self.solve_sewing_constraints_x(compliance_sewing)

    def solve_constraints_v(self):
        self.mesh_dy.verts.dv.fill(0.0)
        self.mesh_dy.verts.nc.fill(0.0)

        if self.enable_collision_handling:
            self.solve_collision_constraints_v(self.mu)

        self.update_dv()

    @ti.kernel
    def update_dx(self):
        for v in self.mesh_dy.verts:
            if v.nc > 0:
                v.y += v.fixed * (v.dx / v.nc)

    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for v in self.mesh_dy.verts:
            if fixed_vertices[v.id] >= 1:
                v.fixed = 0.0
            else:
                v.fixed = 1.0

    @ti.kernel
    def update_dv(self):
        for v in self.mesh_dy.verts:
            if v.nc > 0.0:
                v.dv = v.dv / v.nc

            if v.m_inv > 0.0:
                v.v += v.fixed * v.dv


    def load_sewing_pairs(self):
        data = []

        with open('animation/sewing.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            data = list(reader)

        data_np = np.array(data, dtype=np.int32)
        self.copy_sewing_pairs_to_taichi_field(data_np, len(data_np))
        self.sewing_pairs_num[None] = len(data_np)
        # print(self.sewing_pairs)

    @ti.kernel
    def copy_sewing_pairs_to_taichi_field(self, data_np: ti.types.ndarray(), length: ti.i32):
        for i in range(length):
            self.sewing_pairs[i] = ti.Vector([data_np[i, 0], data_np[i, 1]])

    @ti.kernel
    def solve_sewing_constraints_x(self, compliance: ti.f32):
        for i in range(self.sewing_pairs_num[None]):
            v1_id = self.sewing_pairs[i][0]
            v2_id = self.sewing_pairs[i][1]
            v1_y = self.mesh_dy.verts.y[v1_id]
            v2_y = self.mesh_dy.verts.y[v2_id]

            C = (v2_y - v1_y).norm()
            nabla_C = (v1_y - v2_y).normalized()
            schur = (self.mesh_dy.verts.fixed[v1_id] * self.mesh_dy.verts.m_inv[v1_id] + self.mesh_dy.verts.fixed[v2_id] * self.mesh_dy.verts.m_inv[v2_id])
            ld = compliance * C / (compliance * schur + 1.0)

            self.mesh_dy.verts.dx[v1_id] -= self.mesh_dy.verts.fixed[v1_id] * self.mesh_dy.verts.m_inv[v1_id] * ld * nabla_C
            self.mesh_dy.verts.dx[v2_id] += self.mesh_dy.verts.fixed[v2_id] * self.mesh_dy.verts.m_inv[v2_id] * ld * nabla_C
            self.mesh_dy.verts.nc[v1_id] += 1.0
            self.mesh_dy.verts.nc[v2_id] += 1.0

    def forward(self, n_substeps):

        # self.load_sewing_pairs()
        dt_sub = self.dt / n_substeps
        #
        # if self.enable_collision_handling:
        #
        #     self.mesh_dy.computeAABB_faces(padding=self.padding)
        #     aabb_min_dy, aabb_max_dy = self.mesh_dy.computeAABB(padding=self.padding)
        #     self.lbvh_dy.build(self.mesh_dy, aabb_min_dy, aabb_max_dy)
        #
        #     if self.mesh_st != None:
        #
        #         self.mesh_st.computeAABB_faces(padding=self.padding)
        #         aabb_min_st, aabb_max_st = self.mesh_st.computeAABB(padding=self.padding)
        #         self.lbvh_st.build(self.mesh_st, aabb_min_st, aabb_max_st)

        for _ in range(n_substeps):
            self.compute_y_test(dt_sub)

            # self.compute_y(dt_sub)
            # self.solve_constraints_jacobi_x(dt_sub)
            # self.compute_velocity(damping=self.damping, dt=dt_sub)
            self.compute_velocity_test(damping=self.damping, dt=dt_sub)
            # if self.enable_velocity_update:
            #     self.solve_constraints_v()
            #
            # self.update_x(dt_sub)
