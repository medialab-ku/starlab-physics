import csv
import taichi as ti
import numpy as np
import solve_collision_constraints_x
import solve_collision_constraints_v
from lbvh import LBVH
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

@ti.data_oriented
class Solver:
    def __init__(self,
                 enable_profiler,
                 mesh_dy,
                 mesh_st,
                 grid_size,
                 dHat,
                 YM,
                 PR,
                 g,
                 dt):

        self.mesh_dy = mesh_dy
        self.mesh_st = mesh_st
        self.g = g
        self.dt = dt
        self.dHat = dHat
        self.YM = YM
        self.PR = PR
        self.unit_vector = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.unit_vector[0] = ti.math.vec3(1.0, 0.0, 0.0)
        self.unit_vector[1] = ti.math.vec3(0.0, 1.0, 0.0)
        self.unit_vector[2] = ti.math.vec3(0.0, 0.0, 1.0)
        self.strain_limit = 0.01

        self.rest_volume = ti.field(dtype=ti.f32, shape=1)
        self.rest_volume[0] = 0

        self.current_volume = ti.field(dtype=ti.f32, shape=1)
        self.current_volume[0] = 0.0

        self.num_inverted_elements = ti.field(dtype=ti.i32, shape=1)

        self.grid_size = grid_size
        self.mu = 0.1

        # self.friction_coeff = ti.field(dtype=ti.f32, shape=1)
        # self.friction_coeff[0] = 0
        # self.grid_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
        # self.aabb_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
        # self.grid_edge_indices = ti.field(dtype=ti.u32, shape=12 * 2)
        self.padding = 0.1
        # self.init_grid()

        # self.grid_origin = -self.grid_size
        # self.grid_num = np.ceil(2 * self.grid_size / self.cell_size).astype(int)
        # self.grid_num_dynamic = ti.Vector.field(n=3, dtype=ti.i32, shape=1)
        # print("grid dim:", self.grid_num)

        self.enable_velocity_update = False
        self.enable_collision_handling = False
        self.enable_move_obstacle = False
        self.enable_profiler = enable_profiler
        self.export_mesh = False

        self.obs_lin_vel = ti.Vector.field(n=3, dtype=ti.f32, shape=1)
        self.obs_lin_vel.fill(0.0)
        self.obs_ang_vel = ti.Vector.field(n=3, dtype=ti.f32, shape=1)
        self.obs_ang_vel.fill(0.0)

        self.manual_lin_vels = ti.Vector.field(n=3, dtype=ti.f32, shape=4)
        self.manual_lin_vels.fill(0.0)
        self.manual_ang_vels = ti.Vector.field(n=3, dtype=ti.f32, shape=4)
        self.manual_ang_vels.fill(0.0)

        self.manual_ang_vels[0] = ti.Vector([70.0, 0.0, 0.0])
        self.manual_ang_vels[1] = ti.Vector([-70.0, 0.0, 0.0])
        self.manual_ang_vels[2] = ti.Vector([0, -5.0, 0.0])
        self.manual_ang_vels[3] = ti.Vector([0.0, 1.0, 0.0])


        self.max_num_verts_dy = len(self.mesh_dy.verts)
        self.max_num_edges_dy = len(self.mesh_dy.edges)
        self.max_num_faces_dy = len(self.mesh_dy.faces)

        #
        self.max_num_verts_st = 0
        self.max_num_edges_st = 0
        self.max_num_faces_st = 0

        # self.sorted_id_st = ti.field(dtype=ti.i32, shape=self.max_num_faces_st)
        # self.mesh_st.computeAABB_faces(padding=self.padding)
        # aabb_min_st, aabb_max_st = self.mesh_st.computeAABB()
        self.lbvh_st = None
        if self.mesh_st != None:
            self.lbvh_st = LBVH(len(self.mesh_st.faces))
            self.max_num_verts_st = len(self.mesh_st.verts)
            self.max_num_edges_st = len(self.mesh_st.edges)
            self.max_num_faces_st = len(self.mesh_st.faces)
            self.mesh_st.computeAABB_faces(padding=0.0)
            aabb_min_st, aabb_max_st = self.mesh_st.computeAABB()
            # print(aabb_min_st, aabb_max_st)
            self.lbvh_st.build(self.mesh_st, aabb_min_st, aabb_max_st)

        # print(len(self.mesh_dy.faces))
        self.lbvh_dy = LBVH(len(self.mesh_dy.faces))

        # self.lbvh_st_v = LBVH(len(self.mesh_st.verts))
        # self.lbvh_st.build(self.mesh_st, aabb_min_st, aabb_max_st)
        # print(aabb_min, aabb_max)

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


        # self.ee_static_pair_cache_size = 40
        # self.ee_static_pair = ti.field(dtype=ti.int32, shape=(self.max_num_edges_dynamic, self.ee_static_pair_cache_size, 2))
        # self.ee_static_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_edges_dynamic)
        # self.ee_static_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.ee_static_pair_cache_size, 4))
        # self.ee_static_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.ee_static_pair_cache_size))
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

        self.c_total = ti.field(dtype=ti.f32, shape=1)

        self.is_frame_capture = False
        self.iteration_data = np.array([0])
        self.C_data = np.array([0])

    # @ti.kernel
    # def __init_animation_pos(self, is_selected: ti.template()):
    #     for i in is_selected:
    #         if is_selected != 0:
    #             self.anim_x[i] = self.x[i]
    #
    #     for i in ti.ndrange(4):
    #         count = 0.0
    #         origin = ti.Vector([0.0, 0.0, 0.0])
    #         idx_set_count = i + 1
    #
    #         for pidx in ti.ndrange(self.max_num_verts_dynamic):
    #             if is_selected[pidx] == idx_set_count:
    #                 count += 1
    #                 origin += self.x[pidx]
    #
    #         if count > 0.001:
    #             self.anim_local_origin[i] = origin / count
    #
    # def _reset_animation(self):
    #     self.num_animation.fill(0)
    #     self.cur_animation.fill(0)
    #     self.active_anim_frame.fill(0)
    #     self.action_anim.fill(0.0)
    #     self.anim_local_origin.fill(0.0)
    #     self.anim_rotation_mat.fill(0.0)
    #
    # def _set_animation(self, animationDict, is_selected):
    #     self.num_animation.fill(0)
    #     self.cur_animation.fill(0)
    #     self.active_anim_frame.fill(0)
    #     self.action_anim.fill(0.0)
    #     self.anim_local_origin.fill(0.0)
    #
    #     self.num_animation[0] = len(animationDict[1])
    #     self.num_animation[1] = len(animationDict[2])
    #     self.num_animation[2] = len(animationDict[3])
    #     self.num_animation[3] = len(animationDict[4])
    #     self.num_animation[3] = len(animationDict[4])
    #
    #     if (self.num_animation[0] > self.max_num_anim) or (self.num_animation[1] > self.max_num_anim) \
    #             or (self.num_animation[2] > self.max_num_anim) or (self.num_animation[3] > self.max_num_anim):
    #         print("warning :: length of some animation is longer than ", self.max_num_anim,
    #               ". Subsequent animations are ignored")
    #         self.num_animation[0] = self.num_animation[0] if self.num_animation[
    #                                                              0] < self.max_num_anim else self.max_num_anim
    #         self.num_animation[1] = self.num_animation[1] if self.num_animation[
    #                                                              1] < self.max_num_anim else self.max_num_anim
    #         self.num_animation[2] = self.num_animation[2] if self.num_animation[
    #                                                              2] < self.max_num_anim else self.max_num_anim
    #         self.num_animation[3] = self.num_animation[3] if self.num_animation[
    #                                                              3] < self.max_num_anim else self.max_num_anim
    #
    #     self.__init_animation_pos(is_selected)
    #
    #     for ic in range(4):
    #         animations_ic = animationDict[ic + 1]
    #         for a in range(self.num_animation[ic]):
    #             animation = animations_ic[a]
    #
    #             self.active_anim_frame[ic, a] = animation[6]
    #             for j in range(6):
    #                 self.action_anim[ic, a][j] = animation[j]
    #
    #     self.set_fixed_vertices(is_selected)
    #
    #     # print(self.num_animation)
    #     # print(self.active_anim_frame)
    #     # print(self.action_anim)
    #     # print(self.anim_local_origin)
    #
    #     # self.frame[0] = 0
    #     #
    #     # self.max_num_anim = 40
    #     # self.num_animation = ti.field(dtype = ti.i32,shape = 4)   # maximum number of handle set
    #     # self.anim_local_origin = ti.Vector.field(3, dtype=ti.f32, shape=4)
    #     # self.active_anim_frame = ti.field(dtype = ti.i32,shape = (4,self.max_num_anim)) # maximum number of animation
    #     # self.action_anim = ti.Vector.field(6,dtype = ti.f32,shape = (4,self.max_num_anim)) # maximum number of animation, a animation consist (vx,vy,vz,rx,ry,rz)
    #     #
    #     # self.anim_x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
    #
    # @ti.func
    # def apply_rotation(self, mat_rot, vec):
    #     v = ti.Vector([vec[0], vec[1], vec[2], 1])
    #     vv = mat_rot @ v
    #     return ti.Vector([vv[0], vv[1], vv[2]])
    #
    # @ti.func
    # def get_animation_rotation_mat(self, axis, degree):
    #     s = ti.sin(degree * 0.5)
    #     c = ti.cos(degree * 0.5)
    #     axis = s * axis
    #     q = ti.Vector([axis[0], axis[1], axis[2], c])
    #     M = ti.math.mat4(0.0)
    #
    #     M[3, 3] = 1.0
    #
    #     M[0, 0] = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
    #     M[1, 1] = 1 - 2 * (q[0] ** 2 + q[2] ** 2)
    #     M[2, 2] = 1 - 2 * (q[0] ** 2 + q[1] ** 2)
    #
    #     M[2, 1] = 2 * (q[1] * q[2] + q[3] * q[0])
    #     M[1, 2] = 2 * (q[1] * q[2] - q[3] * q[0])
    #
    #     M[2, 0] = 2 * (q[0] * q[2] - q[3] * q[1])
    #     M[0, 2] = 2 * (q[0] * q[2] + q[3] * q[1])
    #
    #     M[1, 0] = 2 * (q[0] * q[1] + q[3] * q[2])
    #     M[0, 1] = 2 * (q[0] * q[1] - q[3] * q[2])
    #
    #     return M
    #
    # @ti.kernel
    # def animate_handle(self, is_selected: ti.template()):
    #
    #     for idx_set in ti.ndrange(4):
    #         cur_anim = self.cur_animation[idx_set]
    #         max_anim = self.num_animation[idx_set]
    #         is_animation_changed = False or (self.frame[0] == 0)  # when frame = 0 animation changed
    #         while self.active_anim_frame[idx_set, cur_anim] < self.frame[0] and cur_anim < max_anim:
    #             self.cur_animation[idx_set] = self.cur_animation[idx_set] + 1
    #             cur_anim = self.cur_animation[idx_set]
    #             is_animation_changed = True
    #
    #         if cur_anim < max_anim:
    #             vel = self.action_anim[idx_set, cur_anim]
    #             lin_vel = ti.Vector([vel[0], vel[1], vel[2]])
    #             self.anim_local_origin[idx_set] += lin_vel * self.dt[0]
    #
    #             ang_vel = ti.Vector([vel[3], vel[4], vel[5]])
    #             degree_rate = ang_vel.norm()
    #
    #             if is_animation_changed and degree_rate > 1e-4:
    #                 axis = ti.math.normalize(ang_vel)
    #                 self.anim_rotation_mat[idx_set] = self.get_animation_rotation_mat(axis, degree_rate * self.dt[0])
    #
    #     for i in self.anim_x:
    #         if is_selected[i] >= 1:
    #
    #             idx_set = int(is_selected[i] - 1)
    #
    #             cur_anim = int(self.cur_animation[idx_set])
    #             max_anim = int(self.num_animation[idx_set])
    #
    #             if cur_anim < max_anim:
    #                 vel = self.action_anim[idx_set, cur_anim]
    #                 lin_vel = ti.Vector([vel[0], vel[1], vel[2]])
    #                 self.anim_x[i] = self.anim_x[i] + lin_vel * self.dt[0]
    #
    #                 ang_vel = ti.Vector([vel[3], vel[4], vel[5]])
    #                 degree_rate = ang_vel.norm()
    #                 if degree_rate > 1e-4:
    #                     mat_rot = self.anim_rotation_mat[idx_set]
    #
    #                     self.anim_x[i] = self.apply_rotation(mat_rot,
    #                                                          self.anim_x[i] - self.anim_local_origin[idx_set]) + \
    #                                      self.anim_local_origin[idx_set]
    #
    #                 self.x[i] = self.anim_x[i]

    # def init_grid(self):
    #
    #     self.grid_vertices[0] = self.padding * ti.math.vec3(self.grid_size[0], self.grid_size[1], self.grid_size[2])
    #     self.grid_vertices[1] = self.padding * ti.math.vec3(-self.grid_size[0], self.grid_size[1], self.grid_size[2])
    #     self.grid_vertices[2] = self.padding * ti.math.vec3(-self.grid_size[0], self.grid_size[1], -self.grid_size[2])
    #     self.grid_vertices[3] = self.padding * ti.math.vec3(self.grid_size[0], self.grid_size[1], -self.grid_size[2])
    #
    #     self.grid_vertices[4] = self.padding * ti.math.vec3(self.grid_size[0], -self.grid_size[1], self.grid_size[2])
    #     self.grid_vertices[5] = self.padding * ti.math.vec3(-self.grid_size[0], -self.grid_size[1], self.grid_size[2])
    #     self.grid_vertices[6] = self.padding * ti.math.vec3(-self.grid_size[0], -self.grid_size[1], -self.grid_size[2])
    #     self.grid_vertices[7] = self.padding * ti.math.vec3(self.grid_size[0], -self.grid_size[1], -self.grid_size[2])
    #
    #     self.grid_edge_indices[0] = 0
    #     self.grid_edge_indices[1] = 1
    #
    #     self.grid_edge_indices[2] = 1
    #     self.grid_edge_indices[3] = 2
    #
    #     self.grid_edge_indices[4] = 2
    #     self.grid_edge_indices[5] = 3
    #
    #     self.grid_edge_indices[6] = 3
    #     self.grid_edge_indices[7] = 0
    #
    #     self.grid_edge_indices[8] = 4
    #     self.grid_edge_indices[9] = 5
    #
    #     self.grid_edge_indices[10] = 5
    #     self.grid_edge_indices[11] = 6
    #
    #     self.grid_edge_indices[12] = 6
    #     self.grid_edge_indices[13] = 7
    #
    #     self.grid_edge_indices[14] = 7
    #     self.grid_edge_indices[15] = 4
    #
    #     self.grid_edge_indices[16] = 0
    #     self.grid_edge_indices[17] = 4
    #
    #     self.grid_edge_indices[18] = 1
    #     self.grid_edge_indices[19] = 5
    #
    #     self.grid_edge_indices[20] = 2
    #     self.grid_edge_indices[21] = 6
    #
    #     self.grid_edge_indices[22] = 3
    #     self.grid_edge_indices[23] = 7

        # print(self.grid_vertices)
        # print(self.grid_edge_indices)



    def reset(self):
        # self.frame[0] = 0
        self.mesh_dy.reset()

        if self.mesh_st != None:
            self.mesh_st.reset()
            self.mesh_st.computeAABB_faces(padding=0.0)
            aabb_min_st, aabb_max_st = self.mesh_st.computeAABB()
            # print(aabb_min_st, aabb_max_st)
            self.lbvh_st.build(self.mesh_st, aabb_min_st, aabb_max_st)
            # self.broadphase_lbvh()


    @ti.kernel
    def compute_y(self, dt: ti.f32):
        for v in self.mesh_dy.verts:
            v.y = v.x + v.fixed * v.v * dt + self.g * dt * dt



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

    def solve_spring_constraints_x_parallel(self, compliance: ti.f32):
        for i in range(self.mesh_dy.num_colors):
            if i < self.mesh_dy.num_colors - 1:
                current_offset = self.mesh_dy.color_prefix_sum_np[i]
                next_offset = self.mesh_dy.color_prefix_sum_np[i + 1]
                self.parallel_kernel_solver_x(compliance, current_offset, next_offset)
            elif i == self.mesh_dy.num_colors - 1:
                current_offset = self.mesh_dy.color_prefix_sum_np[i]
                next_offset = self.mesh_dy.num_edges
                self.parallel_kernel_solver_x(compliance, current_offset, next_offset)

    @ti.kernel
    def parallel_kernel_solver_x(self, compliance: ti.f32, current_offset: ti.i32, next_offset: ti.i32):
        size = next_offset - current_offset
        for i in range(size):
            edge_idx = self.mesh_dy.sorted_edges_index[i + current_offset]
            v0, v1 = self.mesh_dy.edge_indices[2 * edge_idx + 0], self.mesh_dy.edge_indices[2 * edge_idx + 1]

            l0 = self.mesh_dy.edges.l0[edge_idx]
            x10 = self.mesh_dy.verts.y[v0] - self.mesh_dy.verts.y[v1]
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            schur = (self.mesh_dy.verts.fixed[v0] * self.mesh_dy.verts.m_inv[v0] +
                     self.mesh_dy.verts.fixed[v1] * self.mesh_dy.verts.m_inv[v1]) * nabla_C.dot(nabla_C)
            ld = compliance * C / (compliance * schur + 1.0)

            self.c_total[0] += C

            # we can directly apply constraints to the predicted position of vertex
            # Hence, we don't need to use the update_dx() function
            self.mesh_dy.verts.y[v0] -= self.mesh_dy.verts.fixed[v0] * self.mesh_dy.verts.m_inv[v0] * ld * nabla_C
            self.mesh_dy.verts.y[v1] += self.mesh_dy.verts.fixed[v1] * self.mesh_dy.verts.m_inv[v1] * ld * nabla_C

    @ti.kernel
    def solve_spring_constraints_x(self, compliance: ti.f32):
        # print(compliance)

        # ti.loop_config(serialize=True)
        for e in self.mesh_dy.edges:
            l0 = e.l0
            x10 = e.verts[0].y - e.verts[1].y
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            schur = (e.verts[0].fixed * e.verts[0].m_inv + e.verts[1].fixed * e.verts[1].m_inv) * nabla_C.dot(nabla_C)
            ld = compliance * C / (compliance * schur + 1.0)

            e.verts[0].dx -= e.verts[0].fixed * e.verts[0].m_inv * ld * nabla_C
            e.verts[1].dx += e.verts[1].fixed * e.verts[1].m_inv * ld * nabla_C
            e.verts[0].nc += 1.0
            e.verts[1].nc += 1.0

            self.c_total[0] += C

    @ti.kernel
    def solve_spring_constraints_x_test(self, compliance: ti.f32):

        # print(compliance)
        for e in self.mesh_dy.edges:
            l0 = e.l0
            x10 = e.verts[0].y - e.verts[1].y
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            # ld = C / (compliance * schur + 1.0)
            coef0 = compliance * e.verts[0].fixed * e.verts[0].m_inv
            coef1 = compliance * e.verts[1].fixed * e.verts[1].m_inv
            a, b = coef0 + 1.0, -coef0
            c, d = -coef1, coef1 + 1.0
            det = coef0 + coef1
            g0 = -coef0 * C * nabla_C
            g1 =  coef1 * C * nabla_C
            e.verts[0].dx += (d * g0 - b * g1) / det
            e.verts[1].dx += (-c * g0 + a * g1) / det
            e.verts[0].nc += 1.0
            e.verts[1].nc += 1.0

    @ti.kernel
    def solve_spring_constraints_v(self):
        s = 0
        for ei in range(self.offset_spring):
            if self.spring_cache[ei] > 0:
                ti.atomic_add(s, 1)
                v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
                nablaC = self.spring_cache_g[ei]
                Cv = (self.v[v0] - self.v[v1]).dot(nablaC)
                if Cv > 0.0:
                    ld_v = -Cv / self.spring_cache_schur[ei]
                    self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * nablaC * ld_v
                    self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * nablaC * ld_v
                    self.nc[v0] += 1
                    self.nc[v1] += 1

        # print(s / self.offset_spring)

    @ti.kernel
    def broadphase_lbvh(self) -> ti.int32:

        self.vt_st_candidates_num.fill(0)
        self.tv_st_candidates_num.fill(0)
        self.vt_dy_candidates_num.fill(0)

        cnt = 0
        for i in range(2 * self.max_num_verts_dy + self.max_num_verts_st):
            if i < self.max_num_verts_dy:
                vid = i
                y = self.mesh_dy.verts.y[vid]
                aabb_min = y - self.padding * ti.math.vec3(1.0)
                aabb_max = y + self.padding * ti.math.vec3(1.0)
                a = self.lbvh_st.traverse_bvh_single(aabb_min, aabb_max, vid, self.vt_st_candidates, self.vt_st_candidates_num)
            elif i < 2 * self.max_num_verts_dy:
                vid = i - self.max_num_verts_dy
                # x = self.mesh_dy.verts.x[vid]
                # aabb_min = x - self.padding * ti.math.vec3(1.0)
                # aabb_max = x + self.padding * ti.math.vec3(1.0)
                # a = self.lbvh_dy.traverse_bvh_single(aabb_min, aabb_max, vid, self.vt_dy_candidates, self.vt_dy_candidates_num)
                # ti.atomic_add(cnt, a)
            # else:
            #     vid = i - 2 * self.max_num_verts_dy
            #     x = self.mesh_st.verts.x[vid]
            #     aabb_min = x - self.padding * ti.math.vec3(1.0)
            #     aabb_max = x + self.padding * ti.math.vec3(1.0)
            #     a = self.lbvh_dy.traverse_bvh_single(aabb_min, aabb_max, vid, self.tv_st_candidates, self.tv_st_candidates_num)
            #     ti.atomic_add(cnt, a)

        return cnt

    @ti.kernel
    def broadphase_brute(self) -> ti.int32:
        self.vt_st_candidates_num.fill(0)
        self.vt_dy_candidates_num.fill(0)
        cnt = 0
        for v in self.mesh_dy.verts:
            aabb_min = ti.min(v.x, v.y) - self.padding * ti.math.vec3(1.0)
            aabb_max = ti.max(v.x, v.y) + self.padding * ti.math.vec3(1.0)
            for fi in range(self.max_num_faces_st):
                min1, max1 = self.mesh_st.faces.aabb_min[fi], self.mesh_st.faces.aabb_max[fi]
                # ti.atomic_add(cnt, 1)
                if self.lbvh_st.aabb_overlap(aabb_min, aabb_max, min1, max1):
                    self.vt_st_candidates[v.id, self.vt_st_candidates_num[v.id]] = fi
                    self.vt_st_candidates_num[v.id] += 1

            # for fi in range(self.max_num_faces_dy):
            #     min1, max1 = self.mesh_dy.faces.aabb_min[fi], self.mesh_dy.faces.aabb_max[fi]
            #
            #     if self.lbvh_dy.aabb_overlap(aabb_min, aabb_max, min1, max1):
            #         ti.atomic_add(cnt, 1)
            #         self.vt_dy_candidates[v.id, self.vt_dy_candidates_num[v.id]] = fi
            #         self.vt_dy_candidates_num[v.id] += 1

        return cnt
    @ti.kernel
    def solve_collision_constraints_x(self):

        self.vt_st_pair_num.fill(0)
        self.tv_st_pair_num.fill(0)
        self.vt_dy_pair_num.fill(0)

        d = self.dHat
        for i in range(2 * self.max_num_verts_dy + self.max_num_verts_st):
            if i < self.max_num_verts_dy:
                vid = i
                for j in range(self.vt_st_candidates_num[vid]):
                    fi_s = self.vt_st_candidates[vid, j]
                    solve_collision_constraints_x.__vt_st(vid, fi_s, self.mesh_dy, self.mesh_st, d, self.vt_st_pair_cache_size, self.vt_st_pair, self.vt_st_pair_num, self.vt_st_pair_g, self.vt_st_pair_schur)
            elif i < 2 * self.max_num_verts_dy:
                vid = i - self.max_num_verts_dy
                # for j in range(self.vt_dy_candidates_num[vid]):
                #     fi_d = self.vt_dy_candidates[vid, j]
                #     if self.is_in_face(vid, fi_d) != True:
                #         solve_collision_constraints_x.__vt_dy(vid, fi_d, self.mesh_dy, d, self.vt_st_pair_cache_size, self.vt_dy_pair, self.vt_dy_pair_num, self.vt_dy_pair_g, self.vt_dy_pair_schur)
            else:
                vis = i - 2 * self.max_num_verts_dy
                for j in range(self.tv_st_candidates_num[vis]):
                    fi_d = self.tv_st_candidates[vis, j]
                    solve_collision_constraints_x.__tv_st(fi_d, vis, self.mesh_dy, self.mesh_st, d, self.vt_st_pair_cache_size, self.tv_st_pair, self.tv_st_pair_num, self.tv_st_pair_g, self.tv_st_pair_schur)

    @ti.kernel
    def solve_collision_constraints_v(self, mu: ti.f32):

        for i in range(2 * self.max_num_verts_dy + self.max_num_verts_st):

            if i < self.max_num_verts_dy:
                vid = i
                for j in range(self.vt_st_pair_num[vid]):
                    fi_s, dtype = self.vt_st_pair[vid, j, 0], self.vt_st_pair[vid, j, 1]
                    g0, g1, g2, g3 = self.vt_st_pair_g[vid, j, 0],  self.vt_st_pair_g[vid, j, 1],  self.vt_st_pair_g[vid, j, 2],  self.vt_st_pair_g[vid, j, 3]
                    schur = self.vt_st_pair_schur[vid, j]
                    solve_collision_constraints_v.__vt_st(vid, fi_s, dtype, self.mesh_dy, self.mesh_st, g0, g1, g2, g3, schur, mu)
            elif i < 2 * self.max_num_verts_dy:
                vid = i - self.max_num_verts_dy
                # for j in range(self.vt_dy_pair_num[vid]):
                #     fi_d, dtype = self.vt_dy_pair[vid, j, 0], self.vt_dy_pair[vid, j, 1]
                #     g0, g1, g2, g3 = self.vt_dy_pair_g[vid, j, 0], self.vt_dy_pair_g[vid, j, 1], self.vt_dy_pair_g[vid, j, 2], self.vt_dy_pair_g[vid, j, 3]
                #     schur = self.vt_dy_pair_schur[vid, j]
                #     solve_collision_constraints_v.__vt_dy(vid, fi_d, dtype, self.mesh_dy, g0, g1, g2, g3, schur, mu)
            else:
                vis = i - 2 * self.max_num_verts_dy
                for j in range(self.tv_st_pair_num[vis]):
                    fi_d, dtype = self.tv_st_pair[vis, j, 0], self.tv_st_pair[vis, j, 1]
                    g0, g1, g2, g3 = self.tv_st_pair_g[vis, j, 0],  self.tv_st_pair_g[vis, j, 1],  self.tv_st_pair_g[vis, j, 2],  self.tv_st_pair_g[vis, j, 3]
                    schur = self.tv_st_pair_schur[vis, j]
                    solve_collision_constraints_v.__tv_st(fi_d, vis, dtype, self.mesh_dy, self.mesh_st, g0, g1, g2, g3, schur, mu)

        #
        # for ei_d in range(self.max_num_edges_dynamic):
        #     for j in range(self.ee_static_pair_num[ei_d]):
        #         ei_s, dtype = self.ee_static_pair[ei_d, j, 0], self.ee_static_pair[ei_d, j, 1]
        #         g0, g1, g2, g3 = self.ee_static_pair_g[ei_d, j, 0], self.ee_static_pair_g[ei_d, j, 1], self.ee_static_pair_g[ei_d, j, 2], self.ee_static_pair_g[ei_d, j, 3]
        #         schur = self.ee_static_pair_schur[ei_d, j]
        #         self.solve_collision_ee_static_v(ei_d, ei_s, d, mu)
        #
        #     for j in range(self.ee_dynamic_pair_num[ei_d]):
        #         ej_d, dtype = self.ee_dynamic_pair[ei_d, j, 0], self.ee_dynamic_pair[ei_d, j, 1]
        #         g0, g1, g2, g3 = self.ee_dynamic_pair_g[ei_d, j, 0], self.ee_dynamic_pair_g[ei_d, j, 1], self.ee_dynamic_pair_g[ei_d, j, 2], self.ee_dynamic_pair_g[ei_d, j, 3]
        #         schur = self.ee_dynamic_pair_schur[ei_d, j]
        #         self.solve_collision_ee_dynamic_v(ei_d, ej_d, dtype, g0, g1, g2, g3, schur, mu)


    # @ti.kernel
    # def solve_fem_constraints_x(self, YM: ti.f32, PR: ti.f32):
    #
    #     mu = YM / (2.0 * (1.0 + PR))
    #     la = (YM * PR) / ((1.0 + PR) * (1.0 - 2.0 * PR))
    #
    #     for tid in range(self.max_num_tetra_dynamic):
    #
    #         v0 = self.tetra_indices_dynamic[4 * tid + 0]
    #         v1 = self.tetra_indices_dynamic[4 * tid + 1]
    #         v2 = self.tetra_indices_dynamic[4 * tid + 2]
    #         v3 = self.tetra_indices_dynamic[4 * tid + 3]
    #
    #         x0, x1, x2, x3 = self.y[v0], self.y[v1], self.y[v2], self.y[v3]
    #
    #         x30 = x0 - x3
    #         x31 = x1 - x3
    #         x32 = x2 - x3
    #
    #         Ds = ti.Matrix.cols([x30, x31, x32])
    #
    #         F = Ds @ self.Dm_inv[tid]
    #         U, sig, V = ti.svd(F)
    #         R = U @ V.transpose()
    #
    #         # if U.determinant() < 0:
    #         #     for i in ti.static(range(3)): U[i, 2] *= -1
    #         #     sig[2, 2] = -sig[2, 2]
    #         # if V.determinant() < 0:
    #         #     for i in ti.static(range(3)): V[i, 2] *= -1
    #         #     sig[2, 2] = -sig[2, 2]
    #
    #         H = 2.0 * mu * self.V0[tid] * (F - R) @ self.Dm_inv[tid].transpose()
    #
    #         C = mu * self.V0[tid] * (ti.pow(sig[0, 0] - 1.0, 2) + ti.pow(sig[1, 1] - 1.0, 2) + ti.pow(sig[2, 2] - 1.0, 2))
    #
    #         g0 = ti.Vector([H[j, 0] for j in ti.static(range(3))])
    #         g1 = ti.Vector([H[j, 1] for j in ti.static(range(3))])
    #         g2 = ti.Vector([H[j, 2] for j in ti.static(range(3))])
    #         g3 = -(g0 + g1 + g2)
    #
    #         alpha = 1.0 / (mu * self.V0[tid] * self.dt[0] * self.dt[0])
    #         schur = (self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) +
    #                  self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) +
    #                  self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) +
    #                  self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-3)
    #
    #
    #
    #         ld = -C / schur
    #
    #         self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld
    #         self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * g1 * ld
    #         self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * g2 * ld
    #         self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * g3 * ld
    #
    #         self.nc[v0] += 1
    #         self.nc[v1] += 1
    #         self.nc[v2] += 1
    #         self.nc[v3] += 1
    #
    #
    #         J = sig[0, 0] * sig[1, 1] * sig[2, 2]
    #         J = F.trace()
    #         if sig[0, 0] * sig[1, 1] * sig[2, 2] < 0.0:
    #             ti.atomic_add(self.num_inverted_elements[0], 1)
    #
    #         gamma = 3.
    #         C_vol = 0.5 * la * self.V0[tid] * (J - gamma) * (J - gamma)
    #         H_vol = la * self.V0[tid] * (J - gamma) * self.Dm_inv[tid].transpose()
    #
    #         nabla_C_vol_0 = ti.Vector([H_vol[j, 0] for j in ti.static(range(3))])
    #         nabla_C_vol_1 = ti.Vector([H_vol[j, 1] for j in ti.static(range(3))])
    #         nabla_C_vol_2 = ti.Vector([H_vol[j, 2] for j in ti.static(range(3))])
    #         nabla_C_vol_3 = -(nabla_C_vol_0 + nabla_C_vol_1 + nabla_C_vol_2)
    #         alpha = 1.0 / (la * self.V0[tid] * self.dt[0] * self.dt[0])
    #         schur_vol = (self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0.dot(nabla_C_vol_0) +
    #                     self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1.dot(nabla_C_vol_1) +
    #                     self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2.dot(nabla_C_vol_2) +
    #                     self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3.dot(nabla_C_vol_3) + alpha)
    #
    #         ld_vol = C_vol / schur_vol
    #
    #         self.dx[v0] -= self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0 * ld_vol
    #         self.dx[v1] -= self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1 * ld_vol
    #         self.dx[v2] -= self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2 * ld_vol
    #         self.dx[v3] -= self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3 * ld_vol
    #
    #         self.nc[v0] += 1
    #         self.nc[v1] += 1
    #         self.nc[v2] += 1
    #         self.nc[v3] += 1
    #         #
    #         # ti.atomic_add(self.current_volume[0], vol)
    #
    # @ti.kernel
    # def solve_fem_constraints_v(self, YM: ti.f32, PR: ti.f32):
    #     mu = YM / (2.0 * (1.0 + PR))
    #     la = (YM * PR) / ((1.0 + PR) * (1.0 - 2.0 * PR))
    #
    #     for tid in range(self.max_num_tetra_dynamic):
    #
    #         v0 = self.tetra_indices_dynamic[4 * tid + 0]
    #         v1 = self.tetra_indices_dynamic[4 * tid + 1]
    #         v2 = self.tetra_indices_dynamic[4 * tid + 2]
    #         v3 = self.tetra_indices_dynamic[4 * tid + 3]
    #
    #         x0, x1, x2, x3 = self.y[v0], self.y[v1], self.y[v2], self.y[v3]
    #
    #         x30 = x0 - x3
    #         x31 = x1 - x3
    #         x32 = x2 - x3
    #
    #         Ds = ti.Matrix.cols([x30, x31, x32])
    #
    #         F = Ds @ self.Dm_inv[tid]
    #         U, sig, V = ti.svd(F)
    #         R = U @ V.transpose()
    #
    #         if U.determinant() < 0:
    #             for i in ti.static(range(3)): U[i, 2] *= -1
    #             sig[2, 2] = -sig[2, 2]
    #         if V.determinant() < 0:
    #             for i in ti.static(range(3)): V[i, 2] *= -1
    #             sig[2, 2] = -sig[2, 2]
    #
    #         H = 2.0 * mu * self.V0[tid] * (F - R) @ self.Dm_inv[tid].transpose()
    #
    #         g0 = ti.Vector([H[j, 0] for j in ti.static(range(3))])
    #         g1 = ti.Vector([H[j, 1] for j in ti.static(range(3))])
    #         g2 = ti.Vector([H[j, 2] for j in ti.static(range(3))])
    #         g3 = -(g0 + g1 + g2)
    #
    #         alpha = 1.0 / (mu * self.V0[tid] * self.dt[0] * self.dt[0])
    #         schur = (self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) +
    #                  self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) +
    #                  self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) +
    #                  self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + alpha)
    #
    #         Cv = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
    #
    #         if Cv > 0:
    #             ld = -Cv / schur
    #
    #             self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld
    #             self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * g1 * ld
    #             self.dv[v2] += self.fixed[v2] * self.m_inv[v2] * g2 * ld
    #             self.dv[v3] += self.fixed[v3] * self.m_inv[v3] * g3 * ld
    #
    #             self.nc[v0] += 1
    #             self.nc[v1] += 1
    #             self.nc[v2] += 1
    #             self.nc[v3] += 1
    #
    #         J = sig[0, 0] * sig[1, 1] * sig[2, 2]
    #         J = F.trace()
    #         # if sig[0, 0] * sig[1, 1] * sig[2, 2] < 0.0:
    #         #     ti.atomic_add(self.num_inverted_elements[0], 1)
    #
    #         gamma = 3.0
    #         H_vol = la * self.V0[tid] * (J - gamma) * self.Dm_inv[tid].transpose()
    #
    #         nabla_C_vol_0 = ti.Vector([H_vol[j, 0] for j in ti.static(range(3))])
    #         nabla_C_vol_1 = ti.Vector([H_vol[j, 1] for j in ti.static(range(3))])
    #         nabla_C_vol_2 = ti.Vector([H_vol[j, 2] for j in ti.static(range(3))])
    #         nabla_C_vol_3 = -(nabla_C_vol_0 + nabla_C_vol_1 + nabla_C_vol_2)
    #         alpha = 1.0 / (la * self.V0[tid] * self.dt[0] * self.dt[0])
    #         schur_vol = (self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0.dot(nabla_C_vol_0) +
    #                      self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1.dot(nabla_C_vol_1) +
    #                      self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2.dot(nabla_C_vol_2) +
    #                      self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3.dot(nabla_C_vol_3) + alpha)
    #
    #         Cv = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
    #
    #         if Cv > 0:
    #             ld_vol = -Cv / schur_vol
    #             self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0 * ld_vol
    #             self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1 * ld_vol
    #             self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2 * ld_vol
    #             self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3 * ld_vol
    #
    #             self.nc[v0] += 1
    #             self.nc[v1] += 1
    #             self.nc[v2] += 1
    #             self.nc[v3] += 1


    @ti.kernel
    def update_x(self, dt: ti.f32):

        for v in self.mesh_dy.verts:
            # if v.id != 0:
            v.x += dt * v.dx



    @ti.kernel
    def compute_velocity(self, dt: ti.f32):
        for v in self.mesh_dy.verts:
            v.v = v.fixed * (v.y - v.x) / dt
            v.x += dt * v.v
    def init_variables(self):

        self.mesh_dy.verts.dx.fill(0.0)
        self.mesh_dy.verts.nc.fill(0.0)
        self.c_total.fill(0.0)


    def solve_constraints_jacobi_x(self, dt):

        self.init_variables()
        compliance = self.YM * dt * dt
        self.solve_spring_constraints_x(compliance)
        # self.solve_spring_constraints_x_test(compliance)
        # if self.enable_collision_handling:
        #     cnt_lbvh = self.broadphase_lbvh()
        #     self.solve_collision_constraints_x()

        compliance_sewing = 1000 * self.YM * dt * dt
        # self.solve_sewing_constraints_x(compliance_sewing)
        # self.solve_pressure_constraints_x()
        self.update_dx()

    def solve_constraints_graph_color_x(self, dt):

        self.init_variables()
        compliance = self.YM * dt * dt
        self.solve_spring_constraints_x_parallel(compliance)
        # if self.enable_collision_handling:
        #     cnt_lbvh = self.broadphase_lbvh()
        #     self.solve_collision_constraints_x()

        compliance_sewing = 1000 * self.YM * dt * dt
        # self.solve_sewing_constraints_x(compliance_sewing)
        # self.solve_pressure_constraints_x()

    def solve_constraints_v(self):
        self.mesh_dy.verts.dv.fill(0.0)
        self.mesh_dy.verts.nc.fill(0.0)
        # self.solve_spring_constraints_v()
        if self.enable_collision_handling:
            self.solve_collision_constraints_v(self.mu)
        self.update_dv()

    @ti.kernel
    def update_dx(self):
        for v in self.mesh_dy.verts:
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

    # @ti.kernel
    # def move_static_object(self):
    #
    #     center = ti.math.vec3(0.0)
    #     for i in self.x_static:
    #         center += self.x_static[i]
    #
    #     center /= self.max_num_verts_static
    #
    #     for i in self.x_static:
    #         old = self.x_static[i]
    #         ri = self.x_static[i] - center
    #         v_4d = ti.Vector([ri[0], ri[1], ri[2], 1])
    #         rot_rad = ti.math.radians(self.obs_ang_vel[0] * self.dt[0])
    #         rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
    #
    #         center += self.obs_lin_vel[0] * self.dt[0]
    #
    #         self.x_static[i] = ti.math.vec3(rv[0], rv[1], rv[2]) + center
    #         self.v_static[i] = (self.x_static[i] - old) / self.dt[0]
    #
    #     # for i in self.x_static:
    #     #     x_cur = self.x_static[i]
    #     #     offset = 30
    #     #     if self.frame[0] >= 30:
    #     #         self.v_static[i] = ti.math.vec3(0., 20.0 * ti.math.sin(30. * (self.frame[0] - offset) * self.dt[0]), 0.)
    #     #         self.x_static[i] += self.v_static[i] * self.dt[0]

    # @ti.kernel
    # def move_each_static_object(self, start_idx: ti.template(), len: ti.template(), sid: ti.template()):
    #
    #     center = ti.math.vec3(0.0)
    #     for i in ti.ndrange(len):
    #         center += self.x_static[i + start_idx]
    #
    #     center /= len
    #
    #     for i in ti.ndrange(len):
    #         old = self.x_static[i + start_idx]
    #         ri = self.x_static[i + start_idx] - center
    #         v_4d = ti.Vector([ri[0], ri[1], ri[2], 1])
    #
    #         rot_rad = ti.math.radians(self.manual_ang_vels[sid] * self.dt[0])
    #         rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
    #
    #         center += self.obs_lin_vel[sid] * self.dt[0]
    #
    #         self.x_static[i + start_idx] = ti.math.vec3(rv[0], rv[1], rv[2]) + center
    #         self.v_static[i + start_idx] = (self.x_static[i + start_idx] - old) / self.dt[0]

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

    # @ti.kernel
    # def solve_spring_constraints_x(self, compliance: ti.f32):
    #     # print(compliance)
    #     for e in self.mesh_dy.edges:
    #         l0 = e.l0
    #         x10 = e.verts[0].y - e.verts[1].y
    #         lij = x10.norm()
    #
    #         C = (lij - l0)
    #         nabla_C = x10.normalized()
    #         schur = (e.verts[0].fixed * e.verts[0].m_inv + e.verts[1].fixed * e.verts[1].m_inv) * nabla_C.dot(nabla_C)
    #         ld = compliance * C / (compliance * schur + 1.0)
    #
    #         e.verts[0].dx -= e.verts[0].fixed * e.verts[0].m_inv * ld * nabla_C
    #         e.verts[1].dx += e.verts[1].fixed * e.verts[1].m_inv * ld * nabla_C
    #         e.verts[0].nc += 1.0
    #         e.verts[1].nc += 1.0

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

    def forward(self, n_substeps, frame_num):
        # self.load_sewing_pairs()
        # dt_sub = self.dt / n_substeps
        # ti.profiler.clear_kernel_profiler_info()

        if self.enable_collision_handling:
            self.mesh_dy.computeAABB_faces(padding=self.padding)
            aabb_min_dy, aabb_max_dy = self.mesh_dy.computeAABB()
            # self.lbvh_dy.build(self.mesh_dy, aabb_min_dy, aabb_max_dy)

            if self.mesh_st != None:
                self.mesh_st.computeAABB_faces(padding=self.padding)
                aabb_min_st, aabb_max_st = self.mesh_st.computeAABB()
                self.lbvh_st.build(self.mesh_st, aabb_min_st, aabb_max_st)

            # cnt_lbvh = self.broadphase_lbvh()

        # iterations(to achieve converged solution, x_star)
        dt_sub = self.dt
        for i in range(n_substeps):
            self.compute_y(dt_sub)
            # cnt_brute = self.broadphase_brute()
            # self.solve_constraints_jacobi_x(dt_sub)
            self.solve_constraints_graph_color_x(dt_sub)

            epsilon = 0.001
            # if self.c_total[0] < epsilon:
            #     print(i)
            #     break
            # self.compute_velocity(dt_sub)
            # if self.enable_velocity_update:
            #     self.solve_constraints_v()

            # self.update_x(dt_sub)

            # capture iterations and C convergence per frame
            if self.is_frame_capture:
                self.iteration_data = np.append(self.iteration_data, self.iteration_data[-1] + 1)
                self.C_data = np.append(self.C_data, self.c_total[0])

            # print("brute: ", cnt_brute / self.max_num_verts_dy)
            # print("lbvh:  ", cnt_lbvh  / self.max_num_verts_dy)

        if self.is_frame_capture:
            self.visualize_data(frame_num=frame_num)
            self.is_frame_capture = False
            self.c_total[0] = 0.0

        self.compute_velocity(dt_sub)

    def visualize_data(self, frame_num):
        # print(self.iteration_data)
        # print(self.C_data)
        plt.figure(figsize=(10, 6))
        plt.plot(self.iteration_data, self.C_data, linestyle='-', color='b')
        plt.title('Convergence of C over Iterations (frame : ' + str(frame_num) + ')')
        plt.xlabel('Iteration')
        plt.ylabel('C Value')
        plt.grid(True)

        datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = "./results/statistics/"
        plt.savefig(filepath + datetime_str + "_frame_" + str(frame_num) + ".png")

        plt.show()

        self.iteration_data = np.array([0])
        self.C_data = np.array([0])