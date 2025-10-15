import taichi as ti
import numpy as np

@ti.data_oriented
class NeighborSearch:
    def __init__(self, config, particle_system):
        self.cfg = config
        self.ps = particle_system
        domain_size =  np.array(self.cfg.get_cfg("domainEnd")) - np.array(self.cfg.get_cfg("domainStart"))

        self.dim = len(domain_size)
        assert self.dim > 1

        self.support_radius          = 4.0  * self.cfg.get_cfg("particleRadius")
        self.grid_size               = self.support_radius
        self.grid_num                = np.ceil(domain_size / self.grid_size).astype(int)

        print("grid size: ", self.grid_num)

        self.grid_particles_num      = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.prefix_sum_executor     = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.grid_ids                = ti.field(int, shape=self.ps.particle_max_num)
        self.grid_ids_buffer         = ti.field(int, shape=self.ps.particle_max_num)
        self.grid_ids_new            = ti.field(int, shape=self.ps.particle_max_num)

    @ti.func
    def pos_to_index(self, pos):
        gi = (pos / self.grid_size).cast(int)
        for d in ti.static(range(self.dim)):
            gi[d] = ti.max(0, ti.min(gi[d], self.grid_num[d] - 1))
        return gi

    @ti.func
    def clamp_cell(self, cell):
        out = ti.Vector([0 for _ in range(self.dim)])
        for d in ti.static(range(self.dim)):
            out[d] = ti.max(0, ti.min(cell[d], self.grid_num[d] - 1))
        return out

    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]

    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.kernel
    def update_grid_id(self):

        for i in range(self.ps.particle_num[None]):
            self.ps.cur2ori[i] = i

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0

        for p in range(self.ps.particle_num[None]):
            grid_index       = self.get_flatten_grid_index(self.ps.x[p])
            self.grid_ids[p] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]

    @ti.kernel
    def counting_sort(self):
        n = self.ps.particle_num[None]

        for i in range(n):
            I = n - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset      = self.grid_particles_num[self.grid_ids[I] - 1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]],1) - 1 + base_offset

        for I in range(n):
            new_index                              = self.grid_ids_new[I]
            self.grid_ids_buffer[new_index]        = self.grid_ids[I]
            self.ps.cur2ori_buffer[new_index]      = self.ps.cur2ori[I]
            self.ps.object_id_buffer[new_index]    = self.ps.object_id[I]
            self.ps.x_0_buffer[new_index]          = self.ps.x0[I]
            self.ps.x_buffer[new_index]            = self.ps.x[I]
            self.ps.v_buffer[new_index]            = self.ps.v[I]
            self.ps.acceleration_buffer[new_index] = self.ps.acceleration[I]
            self.ps.m_V_buffer[new_index]          = self.ps.m_V0[I]
            self.ps.m_buffer[new_index]            = self.ps.m[I]
            self.ps.m_inv_buffer[new_index]        = self.ps.m_inv[I]
            self.ps.density_buffer[new_index]      = self.ps.density[I]
            self.ps.density0_buffer[new_index]     = self.ps.density0[I]
            self.ps.pressure_buffer[new_index]     = self.ps.pressure[I]
            self.ps.material_buffer[new_index]     = self.ps.material[I]
            self.ps.color_buffer[new_index]        = self.ps.color[I]
            self.ps.is_dynamic_buffer[new_index]   = self.ps.is_dynamic[I]
            self.ps.n_buffer[new_index]            = self.ps.n[I]

        for I in range(n):
            self.grid_ids[I]        = self.grid_ids_buffer[I]
            self.ps.cur2ori[I]      = self.ps.cur2ori_buffer[I]
            self.ps.object_id[I]    = self.ps.object_id_buffer[I]
            self.ps.x0[I]          = self.ps.x_0_buffer[I]
            self.ps.x[I]            = self.ps.x_buffer[I]
            self.ps.v[I]            = self.ps.v_buffer[I]
            self.ps.acceleration[I] = self.ps.acceleration_buffer[I]
            self.ps.m_V0[I]          = self.ps.m_V_buffer[I]
            self.ps.m[I]            = self.ps.m_buffer[I]
            self.ps.m_inv[I]        = self.ps.m_inv_buffer[I]
            self.ps.density[I]      = self.ps.density_buffer[I]
            self.ps.density0[I]     = self.ps.density0_buffer[I]
            self.ps.pressure[I]     = self.ps.pressure_buffer[I]
            self.ps.material[I]     = self.ps.material_buffer[I]
            self.ps.color[I]        = self.ps.color_buffer[I]
            self.ps.is_dynamic[I]   = self.ps.is_dynamic_buffer[I]
            self.ps.n[I]            = self.ps.n_buffer[I]

        for I in range(n):
            original_idx = self.ps.cur2ori[I]
            self.ps.ori2cur[original_idx] = I

    @ti.kernel
    def narrow_phase(self, x: ti.template()):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.particle_neighbors_num[p_i] = 0
            # self.ps.solid_neighbors_num[p_i] = 0
            center_cell = self.pos_to_index(x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                nbr_cell = self.clamp_cell(center_cell + offset)
                grid_index = self.flatten_grid_index(nbr_cell)
                start = 0
                if grid_index > 0:
                    start = self.grid_particles_num[grid_index - 1]

                end = self.grid_particles_num[grid_index]
                for p_j in range(start, end):
                    # for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                    if p_i != p_j and (x[p_i] - x[p_j]).norm() < self.ps.support_radius:
                        if self.ps.particle_neighbors_num[p_i] < self.ps.cache_size:
                            self.ps.particle_neighbors[p_i, self.ps.particle_neighbors_num[p_i]] = p_j
                            self.ps.particle_neighbors_num[p_i] += 1
                        # if self.ps.material[p_i] == self.ps.material_solid and self.ps.material[p_j] == self.ps.material_solid:
                        #     if self.ps.solid_neighbors_num[p_i] < self.ps.cache_size:
                        #         self.ps.solid_neighbors[p_i, self.ps.solid_neighbors_num[p_i]] = p_j
                        #         self.ps.solid_neighbors_num[p_i] += 1

    @ti.func
    def simulate_collisions(self, p_i, vec):
        c_f = 0.5
        self.ps.v[p_i] -= (1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_3D(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
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

    def broad_phase(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()