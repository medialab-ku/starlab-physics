import taichi as ti
import numpy as np
import distance as di

@ti.data_oriented
class Solver:
    def __init__(self,
                 meshes_dynamic,
                 meshes_static,
                 tet_meshes_dynamic,
                 particles,
                 grid_size,
                 particle_radius,
                 g,
                 dt):

        self.meshes_dynamic = meshes_dynamic
        self.tet_meshes_dynamic = tet_meshes_dynamic
        self.meshes_static = meshes_static
        self.particles = particles
        self.g = g
        self.dt = dt
        self.dHat = 4 * particle_radius * particle_radius
        self.grid_size = grid_size
        self.particle_radius = particle_radius

        self.grid_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
        self.grid_edge_indices = ti.field(dtype=ti.u32, shape=12 * 2)

        self.padding = 0.9
        self.init_grid()

        self.cell_size = 3 * particle_radius
        self.grid_origin = -self.grid_size
        self.grid_num = np.ceil(2 * self.grid_size / self.cell_size).astype(int)

        self.enable_velocity_update = False

        self.max_num_verts_dynamic = 0
        self.max_num_edges_dynamic = 0
        self.max_num_faces_dynamic = 0
        self.max_num_tetra_dynamic = 0

        num_meshes_dynamic = len(self.meshes_dynamic)
        num_tet_meshes_dynamic = len(self.tet_meshes_dynamic)
        num_vert_offsets = num_meshes_dynamic + len(self.particles) + num_tet_meshes_dynamic

        is_verts_dynamic_empty = not bool(num_vert_offsets)
        is_mesh_dynamic_empty = not bool(len(self.meshes_dynamic))
        is_tet_mesh_dynamic_empty = not bool(len(self.tet_meshes_dynamic))

        if is_verts_dynamic_empty is True:
            num_vert_offsets = 1
            self.max_num_verts_dynamic = 1


        if is_mesh_dynamic_empty is True:
            num_meshes_dynamic = 1
            self.max_num_edges_dynamic = 1
            self.max_num_faces_dynamic = 1


        if is_tet_mesh_dynamic_empty is True:
            num_tet_meshes_dynamic = 1
            self.max_num_tetra_dynamic = 1


        self.offset_verts_dynamic = ti.field(int, shape=num_vert_offsets)
        self.offset_edges_dynamic = ti.field(int, shape=num_meshes_dynamic)
        self.offset_faces_dynamic = ti.field(int, shape=num_meshes_dynamic)
        self.offset_tetras_dynamic = ti.field(int, shape=num_tet_meshes_dynamic)

        for mid in range(len(self.meshes_dynamic)):
            self.offset_verts_dynamic[mid] = self.max_num_verts_dynamic
            self.offset_edges_dynamic[mid] = self.max_num_edges_dynamic
            self.offset_faces_dynamic[mid] = self.max_num_faces_dynamic

            self.max_num_verts_dynamic += len(self.meshes_dynamic[mid].verts)
            self.max_num_edges_dynamic += len(self.meshes_dynamic[mid].edges)
            self.max_num_faces_dynamic += len(self.meshes_dynamic[mid].faces)

        self.offset_tet_mesh = self.max_num_verts_dynamic

        for tid in range(len(self.tet_meshes_dynamic)):
            self.offset_verts_dynamic[tid + len(self.meshes_dynamic)] = self.max_num_verts_dynamic
            self.offset_tetras_dynamic[tid] = self.max_num_tetra_dynamic

            self.max_num_verts_dynamic += len(self.tet_meshes_dynamic[tid].verts)
            self.max_num_tetra_dynamic += len(self.tet_meshes_dynamic[tid].cells)


        self.offset_particle = self.max_num_verts_dynamic

        for pid in range(len(self.particles)):
            self.offset_verts_dynamic[pid + len(self.meshes_dynamic) + len(self.tet_meshes_dynamic)] = self.max_num_verts_dynamic
            self.max_num_verts_dynamic += self.particles[pid].num_particles


        print(self.offset_verts_dynamic)
        print(self.offset_tetras_dynamic)
        print(self.max_num_verts_dynamic)
        print(self.max_num_tetra_dynamic)



        self.y = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.dx = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.dv = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.v = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.nc = ti.field(dtype=ti.int32, shape=self.max_num_verts_dynamic)
        self.m_inv = ti.field(dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.Dm_inv = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=self.max_num_tetra_dynamic)

        self.l0 = ti.field(dtype=ti.f32, shape=self.max_num_edges_dynamic)

        self.face_indices_dynamic = ti.field(dtype=ti.i32, shape=3 * self.max_num_faces_dynamic)
        self.edge_indices_dynamic = ti.field(dtype=ti.i32, shape=2 * self.max_num_edges_dynamic)
        self.tetra_indices_dynamic = ti.field(dtype=ti.i32, shape=4 * self.max_num_tetra_dynamic)

        if is_verts_dynamic_empty is True:
            self.max_num_verts_dynamic = 0

        if is_mesh_dynamic_empty is True:
            self.max_num_edges_dynamic = 0
            self.max_num_faces_dynamic = 0

        if is_tet_mesh_dynamic_empty is True:
            self.max_num_tetra_dynamic = 0


        self.max_num_verts_static = 0
        self.max_num_edges_static = 0
        self.max_num_faces_static = 0

        num_meshes_static = len(self.meshes_static)

        self.is_meshes_static_empty = not bool(num_meshes_static)
        if self.is_meshes_static_empty is True:
            num_meshes_static = 1
            self.max_num_verts_static = 1
            self.max_num_edges_static = 1
            self.max_num_faces_static = 1


        self.offset_verts_static = ti.field(int, shape=num_meshes_static)
        self.offset_edges_static = ti.field(int, shape=num_meshes_static)
        self.offset_faces_static = ti.field(int, shape=num_meshes_static)

        if self.is_meshes_static_empty is False:
            for mid in range(num_meshes_static):
                self.offset_verts_static[mid] = self.max_num_verts_static
                self.offset_edges_static[mid] = self.max_num_edges_static
                self.offset_faces_static[mid] = self.max_num_faces_static
                self.max_num_verts_static += len(self.meshes_static[mid].verts)
                self.max_num_edges_static += len(self.meshes_static[mid].edges)
                self.max_num_faces_static += len(self.meshes_static[mid].faces)

        else:
            self.offset_verts_static[0] = 0
            self.offset_edges_static[0] = 0
            self.offset_faces_static[0] = 0

        self.face_indices_static = ti.field(dtype=ti.i32, shape=3 * self.max_num_faces_static)
        self.edge_indices_static = ti.field(dtype=ti.i32, shape=2 * self.max_num_edges_static)

        self.offset_verts_static[len(self.meshes_static)] = self.max_num_verts_static
        self.offset_faces_static[len(self.meshes_static)] = self.max_num_faces_static

        self.x_static = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_static)

        if self.is_meshes_static_empty is True:
            self.max_num_verts_static = 0
            self.max_num_edges_static = 0
            self.max_num_faces_static = 0


        self.init_mesh_aggregation()
        self.init_particle_aggregation()

        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.grid_ids = ti.field(int, shape=self.max_num_verts_dynamic)
        self.grid_ids_buffer = ti.field(int, shape=self.max_num_verts_dynamic)
        self.grid_ids_new = ti.field(int, shape=self.max_num_verts_dynamic)
        self.cur2org = ti.field(int, shape=self.max_num_verts_dynamic)


    def copy_to_meshes(self):
        for mid in range(len(self.meshes_dynamic)):
            self.copy_to_meshes_device(self.offset_verts_dynamic[mid], self.meshes_dynamic[mid])

        for tid in range(len(self.tet_meshes_dynamic)):
            self.copy_to_tet_meshes_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.tet_meshes_dynamic[tid])

    @ti.kernel
    def copy_to_meshes_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            v.x = self.x[offset + v.id]
            v.v = self.v[offset + v.id]

    @ti.kernel
    def copy_to_tet_meshes_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            v.x = self.x[offset + v.id]
            v.v = self.v[offset + v.id]

    def copy_to_particles(self):
        for pid in range(len(self.particles)):
            self.copy_to_particles_device(self.offset_verts_dynamic[pid + len(self.meshes_dynamic) + len(self.tet_meshes_dynamic)], self.particles[pid])

    @ti.kernel
    def copy_to_particles_device(self, offset: ti.int32, particle: ti.template()):
        for i in range(particle.num_particles):
            particle.x[i] = self.x[offset + i]
            particle.v[i] = self.v[offset + i]


    @ti.kernel
    def init_rest_length(self):
        for ei in range(self.max_num_edges_dynamic):
            v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
            x0, x1 = self.x[v0], self.x[v1]
            x10 = x0 - x1
            self.l0[ei] = x10.norm()

    @ti.kernel
    def init_Dm_inv(self):

        for ti in range(self.max_num_tetra_dynamic):
            v0 = self.tetra_indices_dynamic[4 * ti + 0]
            v1 = self.tetra_indices_dynamic[4 * ti + 1]
            v2 = self.tetra_indices_dynamic[4 * ti + 2]
            v3 = self.tetra_indices_dynamic[4 * ti + 3]

            x0, x1, x2, x3 = self.x[v0], self.x[v1], self.x[v2], self.x[v3]

            Dm = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
            self.Dm_inv[ti] = Dm.inverse()



    def init_mesh_aggregation(self):
        for mid in range(len(self.meshes_dynamic)):
            self.init_mesh_quantities_dynamic_device(self.offset_verts_dynamic[mid], self.meshes_dynamic[mid])
            self.init_edge_indices_dynamic_device(self.offset_verts_dynamic[mid], self.offset_edges_dynamic[mid], self.meshes_dynamic[mid])
            self.init_face_indices_dynamic_device(self.offset_verts_dynamic[mid], self.offset_faces_dynamic[mid], self.meshes_dynamic[mid])

        for tid in range(len(self.tet_meshes_dynamic)):
            self.init_tet_mesh_quantities_dynamic_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.tet_meshes_dynamic[tid])
            # self.init_face_indices_dynamic_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.offset_faces_dynamic[tid + len(self.meshes_dynamic)], self.tet_meshes_dynamic[tid])
            self.init_tet_indices_dynamic_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.offset_tetras_dynamic[tid], self.tet_meshes_dynamic[tid])

        for mid in range(len(self.meshes_static)):
            self.init_quantities_static_device(self.offset_verts_static[mid], self.meshes_static[mid])
            self.init_edge_indices_static_device(self.offset_verts_static[mid], self.offset_edges_static[mid], self.meshes_static[mid])
            self.init_face_indices_static_device(self.offset_verts_static[mid], self.offset_faces_static[mid], self.meshes_static[mid])

        self.init_rest_length()
        self.init_Dm_inv()


    @ti.kernel
    def init_mesh_quantities_dynamic_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            self.x[offset + v.id] = v.x
            self.v[offset + v.id] = v.v
            self.m_inv[offset + v.id] = v.m_inv

    @ti.kernel
    def init_tet_mesh_quantities_dynamic_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            self.x[offset + v.id] = v.x
            self.v[offset + v.id] = v.v
            self.m_inv[offset + v.id] = v.m_inv


    @ti.kernel
    def init_quantities_static_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            self.x_static[offset + v.id] = v.x


    @ti.kernel
    def init_edge_indices_dynamic_device(self, offset_verts: ti.int32, offset_edges: ti.int32, mesh: ti.template()):
        for e in mesh.edges:
            self.edge_indices_dynamic[2 * (offset_edges + e.id) + 0] = e.verts[0].id + offset_verts
            self.edge_indices_dynamic[2 * (offset_edges + e.id) + 1] = e.verts[1].id + offset_verts

    @ti.kernel
    def init_face_indices_dynamic_device(self, offset_verts: ti.int32, offset_faces: ti.int32, mesh: ti.template()):
        for f in mesh.faces:
            self.face_indices_dynamic[3 * (offset_faces + f.id) + 0] = f.verts[0].id + offset_verts
            self.face_indices_dynamic[3 * (offset_faces + f.id) + 1] = f.verts[1].id + offset_verts
            self.face_indices_dynamic[3 * (offset_faces + f.id) + 2] = f.verts[2].id + offset_verts

    @ti.kernel
    def init_tet_indices_dynamic_device(self, offset_verts: ti.int32, offset_tets: ti.int32, mesh: ti.template()):
        for c in mesh.cells:
            self.tetra_indices_dynamic[4 * (offset_tets + c.id) + 0] = c.verts[0].id + offset_verts
            self.tetra_indices_dynamic[4 * (offset_tets + c.id) + 1] = c.verts[1].id + offset_verts
            self.tetra_indices_dynamic[4 * (offset_tets + c.id) + 2] = c.verts[2].id + offset_verts
            self.tetra_indices_dynamic[4 * (offset_tets + c.id) + 3] = c.verts[3].id + offset_verts


    @ti.kernel
    def init_edge_indices_static_device(self, offset_verts: ti.int32, offset_edges: ti.int32, mesh: ti.template()):
        for e in mesh.edges:
            self.edge_indices_static[2 * (offset_edges + e.id) + 0] = e.verts[0].id + offset_verts
            self.edge_indices_static[2 * (offset_edges + e.id) + 1] = e.verts[1].id + offset_verts

    @ti.kernel
    def init_face_indices_static_device(self, offset_verts: ti.int32, offset_faces: ti.int32, mesh: ti.template()):
        for f in mesh.faces:
            self.face_indices_static[3 * (offset_faces + f.id) + 0] = f.verts[0].id + offset_verts
            self.face_indices_static[3 * (offset_faces + f.id) + 1] = f.verts[1].id + offset_verts
            self.face_indices_static[3 * (offset_faces + f.id) + 2] = f.verts[2].id + offset_verts


    def init_particle_aggregation(self):
        for pid in range(len(self.particles)):
            self.init_particle_quantities_dynamic_device(self.offset_verts_dynamic[pid + len(self.meshes_dynamic) + len(self.tet_meshes_dynamic)], self.particles[pid])

    @ti.kernel
    def init_particle_quantities_dynamic_device(self, offset: ti.int32, particle: ti.template()):
        for i in range(particle.num_particles):
            self.x[offset + i] = particle.x[i]
            self.v[offset + i] = particle.v[i]
            self.m_inv[offset + i] = particle.m_inv[i]

    def init_grid(self):

        self.grid_vertices[0] = self.padding * ti.math.vec3(self.grid_size[0], self.grid_size[1], self.grid_size[2])
        self.grid_vertices[1] = self.padding * ti.math.vec3(-self.grid_size[0], self.grid_size[1], self.grid_size[2])
        self.grid_vertices[2] = self.padding * ti.math.vec3(-self.grid_size[0], self.grid_size[1], -self.grid_size[2])
        self.grid_vertices[3] = self.padding * ti.math.vec3(self.grid_size[0], self.grid_size[1], -self.grid_size[2])

        self.grid_vertices[4] = self.padding * ti.math.vec3(self.grid_size[0], -self.grid_size[1], self.grid_size[2])
        self.grid_vertices[5] = self.padding * ti.math.vec3(-self.grid_size[0], -self.grid_size[1], self.grid_size[2])
        self.grid_vertices[6] = self.padding * ti.math.vec3(-self.grid_size[0], -self.grid_size[1], -self.grid_size[2])
        self.grid_vertices[7] = self.padding * ti.math.vec3(self.grid_size[0], -self.grid_size[1], -self.grid_size[2])

        self.grid_edge_indices[0] = 0
        self.grid_edge_indices[1] = 1

        self.grid_edge_indices[2] = 1
        self.grid_edge_indices[3] = 2

        self.grid_edge_indices[4] = 2
        self.grid_edge_indices[5] = 3

        self.grid_edge_indices[6] = 3
        self.grid_edge_indices[7] = 0

        self.grid_edge_indices[8] = 4
        self.grid_edge_indices[9] = 5

        self.grid_edge_indices[10] = 5
        self.grid_edge_indices[11] = 6

        self.grid_edge_indices[12] = 6
        self.grid_edge_indices[13] = 7

        self.grid_edge_indices[14] = 7
        self.grid_edge_indices[15] = 4

        self.grid_edge_indices[16] = 0
        self.grid_edge_indices[17] = 4

        self.grid_edge_indices[18] = 1
        self.grid_edge_indices[19] = 5

        self.grid_edge_indices[20] = 2
        self.grid_edge_indices[21] = 6

        self.grid_edge_indices[22] = 3
        self.grid_edge_indices[23] = 7

        # print(self.grid_vertices)
        # print(self.grid_edge_indices)



    def reset(self):
        for mid in range(len(self.meshes_dynamic)):
            self.meshes_dynamic[mid].reset()

        for pid in range(len(self.particles)):
            self.particles[pid].reset()

        for tid in range(len(self.tet_meshes_dynamic)):
            self.tet_meshes_dynamic[tid].reset()

        self.init_mesh_aggregation()
        self.init_particle_aggregation()

    @ti.kernel
    def compute_y(self):

        # self.m_inv[0] = 0.0
        for i in range(self.max_num_verts_dynamic):
                self.y[i] = self.x[i] + self.dt * self.v[i] + self.g * self.dt * self.dt


    @ti.kernel
    def compute_y_particle(self, particle: ti.template()):
        for i in range(particle.num_particles):
            particle.y[i] = particle.x[i] + particle.v[i] * self.dt + self.g * self.dt * self.dt

    @ti.kernel
    def counting_sort(self):
        # FIXME: make it the actual particle num
        for i in range(self.max_num_verts_dynamic):
            I = self.max_num_verts_dynamic - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I] - 1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

        for i in self.grid_ids:
            new_index = self.grid_ids_new[i]
            self.cur2org[new_index] = i

    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]

    @ti.func
    def pos_to_index(self, pos):
        return ((pos - self.grid_origin) / self.cell_size).cast(int)

    @ti.func
    def get_flatten_grid_index(self, pos: ti.math.vec3):
        return self.flatten_grid_index(self.pos_to_index(pos))




    def broad_phase(self):

        # self.grid_particles_num.fill(0)
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()

    @ti.kernel
    def update_grid_id(self):

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0

        #TODO: update the following two for-loops into a single one
        for vi in range(self.max_num_verts_dynamic):
            grid_index = self.get_flatten_grid_index(self.y[vi])
            self.grid_ids[vi] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]


    @ti.func
    def is_in_face(self, vid, fid):

        v1 = self.face_indices_dynamic[3 * fid + 0]
        v2 = self.face_indices_dynamic[3 * fid + 1]
        v3 = self.face_indices_dynamic[3 * fid + 2]

        return (v1 == vid) or (v2 == vid) or (v3 == vid)

    @ti.func
    def solve_collision_vv(self, v0, v1):

        if v0 != v1:
            x0 = self.y[v0]
            x1 = self.y[v1]
            #
            # x01 = x1 - x0
            #
            # dist = ti.length(x01)
            # nor = x01 / dist
            #
            d = di.d_PP(x0, x1)

            g0, g1 = di.g_PP(x0, x1)
            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (self.dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1

                self.nc[v0] += 1
                self.nc[v1] += 1




    @ti.func
    def solve_collision_vt_dynamic_x(self, vid, fid, dHat):

        v0 = vid
        v1 = self.face_indices_dynamic[3 * fid + 0]
        v2 = self.face_indices_dynamic[3 * fid + 1]
        v3 = self.face_indices_dynamic[3 * fid + 2]

        x0 = self.y[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v2] += self.m_inv[v2] * ld * g2

                self.nc[v0] += 1
                self.nc[v2] += 1

        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v3] += self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v3] += 1

        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.dx[v2] += self.m_inv[v2] * ld * g2

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1

        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v2] += self.m_inv[v2] * ld * g2
                self.dx[v3] += self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.dx[v3] += self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)

            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                    schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                    ld = (dHat - d) / schur
                    # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                    self.dx[v0] += self.m_inv[v0] * ld * g0
                    self.dx[v1] += self.m_inv[v1] * ld * g1
                    self.dx[v2] += self.m_inv[v2] * ld * g2
                    self.dx[v3] += self.m_inv[v3] * ld * g3
                    self.nc[v0] += 1
                    self.nc[v1] += 1
                    self.nc[v2] += 1
                    self.nc[v3] += 1


    @ti.func
    def solve_collision_vt_static_x(self, vid_d, fid_s, dHat):

        v0 = vid_d
        v1 = self.face_indices_static[3 * fid_s + 0]
        v2 = self.face_indices_static[3 * fid_s + 1]
        v3 = self.face_indices_static[3 * fid_s + 2]

        x0 = self.y[v0]
        x1 = self.x_static[v1]
        x2 = self.x_static[v2]
        x3 = self.x_static[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.nc[v0] += 1
                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.nc[v0] += 1


        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)

            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

    @ti.func
    def solve_collision_tv_static_x(self, fid_d, vid_s, dHat):

        v0 = vid_s
        v1 = self.face_indices_dynamic[3 * fid_d + 0]
        v2 = self.face_indices_dynamic[3 * fid_d + 1]
        v3 = self.face_indices_dynamic[3 * fid_d + 2]

        x0 = self.x_static[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)

                self.dx[v1] += self.m_inv[v1] * ld * g1

                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)

                self.dx[v2] += self.m_inv[v2] * ld * g2


                self.nc[v2] += 1

        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)

                self.dx[v3] += self.m_inv[v3] * ld * g3

                self.nc[v3] += 1

        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)

                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.dx[v2] += self.m_inv[v2] * ld * g2

                self.nc[v1] += 1
                self.nc[v2] += 1


        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)

                self.dx[v2] += self.m_inv[v2] * ld * g2
                self.dx[v3] += self.m_inv[v3] * ld * g3

                self.nc[v2] += 1
                self.nc[v3] += 1


        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v1] * g1.dot(g1) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)

                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.dx[v3] += self.m_inv[v3] * ld * g3

                self.nc[v1] += 1
                self.nc[v3] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)

            if d < dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)

                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.dx[v2] += self.m_inv[v2] * ld * g2
                self.dx[v3] += self.m_inv[v3] * ld * g3

                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

    @ti.func
    def solve_collision_vt_static_v(self, vid_d, fid_s, dHat):

        v0 = vid_d
        v1 = self.face_indices_dynamic[3 * fid_s + 0]
        v2 = self.face_indices_dynamic[3 * fid_s + 1]
        v3 = self.face_indices_dynamic[3 * fid_s + 2]

        x0 = self.y[v0]
        x1 = self.x_static[v1]
        x2 = self.x_static[v2]
        x3 = self.x_static[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.nc[v0] += 1


        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            dvn = g0.dot(self.v[v0]) + g2.dot(self.v[v2])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.nc[v0] += 1


        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            dvn = g0.dot(self.v[v0]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.nc[v0] += 1


        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            dvn = g0.dot(self.v[v0]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.nc[v0] += 1


        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
            if d <= dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

    @ti.func
    def solve_collision_vt_dynamic_v(self, vid_d, fid_d, dHat):

        v0 = vid_d
        v1 = self.face_indices_dynamic[3 * fid_d + 0]
        v2 = self.face_indices_dynamic[3 * fid_d + 1]
        v3 = self.face_indices_dynamic[3 * fid_d + 2]

        x0 = self.y[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v1] -= self.m_inv[v1] * ld * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            dvn = g0.dot(self.v[v0]) + g2.dot(self.v[v2])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v2] * g2.dot(g2) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v2] -= self.m_inv[v1] * ld * g2

                self.nc[v0] += 1
                self.nc[v2] += 1


        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            dvn = g0.dot(self.v[v0]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v3] -= self.m_inv[v1] * ld * g3

                self.nc[v0] += 1
                self.nc[v3] += 1


        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v1] -= self.m_inv[v0] * ld * g1
                self.dv[v2] -= self.m_inv[v1] * ld * g2

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1

        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            dvn = g0.dot(self.v[v0]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v2] -= self.m_inv[v2] * ld * g2
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1


        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
            if d <= dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + \
                        self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.dv[v2] -= self.m_inv[v2] * ld * g2
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

    @ti.func
    def solve_collision_tv_static_v(self, vid_s, fid_d, dHat):
        v0 = vid_s
        v1 = self.face_indices_dynamic[3 * fid_d + 0]
        v2 = self.face_indices_dynamic[3 * fid_d + 1]
        v3 = self.face_indices_dynamic[3 * fid_d + 2]

        x0 = self.x_static[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v1] * g1.dot(g1) + 1e-6
                ld = dvn / schur

                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            dvn = g0.dot(self.v[v0]) + g2.dot(self.v[v2])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v2] * g2.dot(g2) + 1e-6
                ld = dvn / schur

                self.dv[v2] -= self.m_inv[v1] * ld * g2
                self.nc[v2] += 1


        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            dvn = g0.dot(self.v[v0]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur

                self.dv[v3] -= self.m_inv[v1] * ld * g3
                self.nc[v3] += 1


        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + 1e-6
                ld = dvn / schur

                self.dv[v1] -= self.m_inv[v0] * ld * g1
                self.dv[v2] -= self.m_inv[v1] * ld * g2

                self.nc[v1] += 1
                self.nc[v2] += 1

        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            dvn = g0.dot(self.v[v0]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur

                self.dv[v2] -= self.m_inv[v2] * ld * g2
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v2] += 1
                self.nc[v3] += 1


        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v[v3])
            if d < dHat and dvn < 0.0:
                schur = self.m_inv[v1] * g1.dot(g1) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur

                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v1] += 1
                self.nc[v3] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
            dvn = g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
            if d <= dHat and dvn < 0.0:
                schur = self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur

                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.dv[v2] -= self.m_inv[v2] * ld * g2
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

    @ti.kernel
    def solve_spring_constraints_x(self):

        for ei in range(self.max_num_edges_dynamic):
            v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
            x0, x1 = self.y[v0], self.y[v1]
            l0 = self.l0[ei]

            x10 = x0 - x1
            center = 0.5 * (x0 + x1)
            lij = x10.norm()
            normal = x10 / lij
            p0 = center + 0.5 * l0 * normal
            p1 = center - 0.5 * l0 * normal

            self.dx[v0] += (p0 - x0)
            self.dx[v1] += (p1 - x1)
            self.nc[v0] += 1
            self.nc[v1] += 1

    @ti.func
    def spiky_gradient(self, r, h):
        result = ti.math.vec3(0.0)

        spiky_grad_factor = -45.0 / ti.math.pi
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        poly6_factor = 315.0 / 64.0 / ti.math.pi
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = poly6_factor * x * x * x
        return result

    @ti.kernel
    def solve_pressure_constraints_x(self):

        kernel_radius = 1.5 * self.particle_radius

        for vi in range(self.max_num_verts_dynamic):
            C_i = self.poly6_value(0.0, kernel_radius) - 1.0
            nabla_C_ii = ti.math.vec3(0.0)
            schur = 1e-4
            xi = self.y[vi]

            # for vj in range(self.max_num_verts_dynamic):
            center_cell = self.pos_to_index(self.y[vi])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
                grid_index = self.flatten_grid_index(center_cell + offset)
                for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                    vj = self.cur2org[p_j]
                    xj = self.y[vj]
                    xji = xj - xi

                    if xji.norm() < kernel_radius:
                        nabla_C_ji = self.spiky_gradient(xji, kernel_radius)
                        C_i += self.poly6_value(xji.norm(), kernel_radius)
                        nabla_C_ii -= nabla_C_ji
                        schur += nabla_C_ji.dot(nabla_C_ji)

                if C_i < 0.0:
                    C_i = 0.0

            schur += nabla_C_ii.dot(nabla_C_ii)
            lambda_i = C_i / schur

            # for vj in range(self.max_num_verts_dynamic):
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
                grid_index = self.flatten_grid_index(center_cell + offset)
                for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                    vj = self.cur2org[p_j]
                    xj = self.y[vj]
                    xji = xj - xi

                    if xji.norm() < kernel_radius:
                        nabla_C_ji = self.spiky_gradient(xji, kernel_radius)
                        self.dx[vj] -= lambda_i * nabla_C_ji
                        self.nc[vj] += 1

            self.dx[vi] -= lambda_i * nabla_C_ii
            self.nc[vi] += 1

    @ti.kernel
    def solve_pressure_constraints_v(self):

        kernel_radius = 2.0 * self.particle_radius

        for vi in range(self.max_num_verts_dynamic):
            C_i = self.poly6_value(0.0, kernel_radius) - 1.0
            C_i_v = 0.0
            nabla_C_ii = ti.math.vec3(0.0)
            schur = 1e-4
            xi = self.y[vi]
            v_i = self.v[vi]

            for vj in range(self.max_num_verts_dynamic):
                # center_cell = self.pos_to_index(self.y[vi])
                # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
                #     grid_index = self.flatten_grid_index(center_cell + offset)
                #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                #         vj = self.cur2org[p_j]
                xj = self.y[vj]
                v_j = self.v[vj]
                xji = xj - xi

                if xji.norm() < kernel_radius:
                    nabla_C_ji = self.spiky_gradient(xji, kernel_radius)
                    C_i_v += nabla_C_ji.dot(v_j)
                    C_i += self.poly6_value(xji.norm(), kernel_radius)
                    nabla_C_ii -= nabla_C_ji
                    schur += nabla_C_ji.dot(nabla_C_ji)


            lambda_i = 0.
            schur += nabla_C_ii.dot(nabla_C_ii)
            C_i_v += nabla_C_ii.dot(v_i)
            if C_i < 0.0 and C_i_v <0:
                lambda_i = C_i_v / schur




            for vj in range(self.max_num_verts_dynamic):
                # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
                #     grid_index = self.flatten_grid_index(center_cell + offset)
                #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                #         vj = self.cur2org[p_j]
                xj = self.y[vj]
                xji = xj - xi

                if xji.norm() < kernel_radius:
                    nabla_C_ji = self.spiky_gradient(xji, kernel_radius)
                    self.dv[vj] -= lambda_i * nabla_C_ji
                    self.nc[vj] += 1

            self.dv[vi] -= lambda_i * nabla_C_ii
            self.nc[vi] += 1


    @ti.kernel
    def solve_collision_constraints_x(self):

        for vi_d in range(self.max_num_verts_dynamic):
            # center_cell = self.pos_to_index(self.y[vi])
            # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            #     grid_index = self.flatten_grid_index(center_cell)
            #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
            #         vj = self.cur2org[p_j]r
            #     # for vj in range(self.max_num_verts):
            #         self.solve_collision_vv(vi, vj)
                    # if self.is_in_face(vi, fi) != True:
                    #     self.solve_collision_vt(vi, fi)
            d = self.dHat

            if vi_d >= self.offset_particle:
                d = ti.pow(self.particle_radius + ti.sqrt(self.dHat), 2)

            for fi_s in range(self.max_num_faces_static):
                self.solve_collision_vt_static_x(vi_d, fi_s, d)

            for fi_d in range(self.max_num_faces_dynamic):
                if self.is_in_face(vi_d, fi_d) != True:
                    self.solve_collision_vt_dynamic_x(vi_d, fi_d, d)

        for fi_d in range(self.max_num_faces_dynamic):
            for vi_s in range(self.max_num_verts_static):
                # if self.is_in_face(vi_d, fi_s) != True:
                self.solve_collision_tv_static_x(fi_d, vi_s, self.dHat)


    @ti.kernel
    def solve_collision_constraints_v(self):
        for vi_d in range(self.max_num_verts_dynamic):
            # center_cell = self.pos_to_index(self.y[vi])
            # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            #     grid_index = self.flatten_grid_index(center_cell)
            #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
            #         vj = self.cur2org[p_j]r
            #     # for vj in range(self.max_num_verts):v
            #         self.solve_collision_vv(vi, vj)
            # if self.is_in_face(vi, fi) != True:
            #     self.solve_collision_vt(vi, fi)
            d = self.dHat

            if vi_d >= self.offset_particle:
                d = ti.pow(self.particle_radius + ti.sqrt(self.dHat), 2)

            for fi_s in range(self.max_num_faces_static):
                self.solve_collision_vt_static_v(vi_d, fi_s, d)

            for fi_d in range(self.max_num_faces_dynamic):
                if self.is_in_face(vi_d, fi_d) != True:
                    self.solve_collision_vt_dynamic_v(vi_d, fi_d, d)

        for fi_d in range(self.max_num_faces_dynamic):
            for vi_s in range(self.max_num_verts_static):
                # if self.is_in_face(vi_d, fi_s) != True:
                self.solve_collision_tv_static_x(fi_d, vi_s, self.dHat)


    @ti.kernel
    def solve_stretch_constarints_x(self):

        for tid in range(self.max_num_tetra_dynamic):

            v0 = self.tetra_indices_dynamic[4 * tid + 0]
            v1 = self.tetra_indices_dynamic[4 * tid + 1]
            v2 = self.tetra_indices_dynamic[4 * tid + 2]
            v3 = self.tetra_indices_dynamic[4 * tid + 3]

            x0, x1, x2, x3 = self.y[v0], self.y[v1], self.y[v2], self.y[v3]
            Ds = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])

            F = Ds @ self.Dm_inv[tid]
            U, sig, V = ti.svd(F)
            R = U @ V.transpose()

            H = (F - R) @ self.Dm_inv[tid].transpose()

            C = 0.5 * (F - R).norm() * (F - R).norm()

            nabla_C0 = ti.Vector([H[j, 0] for j in ti.static(range(3))])
            nabla_C1 = ti.Vector([H[j, 1] for j in ti.static(range(3))])
            nabla_C2 = ti.Vector([H[j, 2] for j in ti.static(range(3))])
            nabla_C3 = -(nabla_C0 + nabla_C1 + nabla_C2)

            schur = (self.m_inv[v0] * nabla_C0.dot(nabla_C0) +
                     self.m_inv[v1] * nabla_C1.dot(nabla_C1) +
                     self.m_inv[v2] * nabla_C2.dot(nabla_C2) +
                     self.m_inv[v3] * nabla_C3.dot(nabla_C3) + 1e-4)

            ld = C / schur

            self.dx[v0] -= self.m_inv[v0] * nabla_C0 * ld
            self.dx[v1] -= self.m_inv[v1] * nabla_C1 * ld
            self.dx[v2] -= self.m_inv[v2] * nabla_C2 * ld
            self.dx[v3] -= self.m_inv[v3] * nabla_C3 * ld


            self.nc[v0] += 1
            self.nc[v1] += 1
            self.nc[v2] += 1
            self.nc[v3] += 1


    @ti.kernel
    def update_x(self):

        for i in range(self.max_num_verts_dynamic):
                self.x[i] += self.v[i] * self.dt

    @ti.kernel
    def update_x_and_v_particle(self, particle: ti.template()):
        for i in range(particle.num_particles):
            particle.v[i] = (particle.y[i] - particle.x[i]) / self.dt
            particle.x[i] = particle.y[i]


    @ti.kernel
    def confine_to_boundary(self):


        for vi in range(self.max_num_verts_dynamic):

            if self.y[vi][0] > self.padding * self.grid_size[0]:
                self.y[vi][0] = self.padding * self.grid_size[0]

            if self.y[vi][0] < -self.padding * self.grid_size[0]:
                self.y[vi][0] = -self.padding * self.grid_size[0]

            if self.y[vi][1] > self.padding * self.grid_size[1]:
                self.y[vi][1] = self.padding * self.grid_size[1]

            if self.y[vi][1] < -self.padding * self.grid_size[2]:
                self.y[vi][1] = -self.padding * self.grid_size[2]

            if self.y[vi][2] > self.padding * self.grid_size[2]:
                self.y[vi][2] = self.padding * self.grid_size[2]

            if self.y[vi][2] < -self.padding * self.grid_size[2]:
                self.y[vi][2] = -self.padding * self.grid_size[2]

    @ti.kernel
    def confine_to_boundary_v(self):

        for vi in range(self.max_num_verts_dynamic):

            if self.y[vi][0] > self.grid_size[0] and self.v[vi][0] > 0:
                self.v[vi][0] = 0

            if self.y[vi][0] < -self.grid_size[0] and self.v[vi][0] < 0:
                self.v[vi][0] = 0

            if self.y[vi][1] > self.grid_size[1] and self.v[vi][1] > 0:
                self.v[vi][1] = 0

            if self.y[vi][1] < -self.grid_size[1] and self.v[vi][1] < 0:
                self.v[vi][1] = 0

            if self.y[vi][2] > self.grid_size[2] and self.v[vi][2] > 0:
                self.v[vi][2] = 0

            if self.y[vi][2] < -self.grid_size[2] and self.v[vi][2] < 0:
                self.v[vi][2] = 0


    @ti.kernel
    def compute_velocity(self):
        for i in range(self.max_num_verts_dynamic):
                self.v[i] = (self.y[i] - self.x[i]) / self.dt



    def solve_constraints_x(self):

        self.dx.fill(0.0)
        self.nc.fill(0)
        self.solve_spring_constraints_x()
        # self.solve_collision_constraints_x()
        self.solve_stretch_constarints_x()
        self.solve_pressure_constraints_x()
        self.update_dx()

    def solve_constraints_v(self):
        self.dv.fill(0.0)
        self.nc.fill(0)
        # self.solve_collision_constraints_v()
        self.solve_pressure_constraints_v()
        self.update_dv()

    @ti.kernel
    def update_dx(self):
        for vi in range(self.max_num_verts_dynamic):
            if self.nc[vi] > 0.0:
                self.dx[vi] = self.dx[vi] / self.nc[vi]
                self.y[vi] += self.dx[vi]


    @ti.kernel
    def update_dv(self):
        for vi in range(self.max_num_verts_dynamic):
            if self.nc[vi] > 0.0:
                self.dv[vi] = self.dv[vi] / self.nc[vi]

            if self.m_inv[vi] > 0.0:
                self.v[vi] += self.dv[vi]

    def forward(self, n_substeps):

        dt = self.dt
        self.dt = dt / n_substeps

        self.broad_phase()
        for _ in range(n_substeps):
            self.compute_y()
            self.solve_constraints_x()
            self.confine_to_boundary()
            self.compute_velocity()

            if self.enable_velocity_update:
                self.solve_constraints_v()
                self.confine_to_boundary_v()
            self.update_x()

        self.copy_to_meshes()
        self.copy_to_particles()


        self.dt = dt
