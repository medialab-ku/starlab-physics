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
    def update_grid_id(self, ps: ti.template()):

        for I in ti.grouped(ps.grid_particles_num):
            ps.grid_particles_num[I] = 0

        for p in range(ps.particle_num[None]):
            grid_index      = self.get_flatten_grid_index(ps.x[p])
            ps.grid_ids[p]  = grid_index
            ti.atomic_add(ps.grid_particles_num[grid_index], 1)

        for I in ti.grouped(ps.grid_particles_num):
            ps.grid_particles_num_temp[I] = ps.grid_particles_num[I]

    @ti.kernel
    def counting_sort(self, ps: ti.template()):
        n = ps.particle_num[None]

        for i in range(n):
            I = n - 1 - i
            base_offset = 0
            if ps.grid_ids[I] - 1 >= 0:
                base_offset     = ps.grid_particles_num[ps.grid_ids[I] - 1]
            ps.grid_ids_new[I]  = ti.atomic_sub(ps.grid_particles_num_temp[ps.grid_ids[I]],1) - 1 + base_offset

        for I in range(n):
            new_index = ps.grid_ids_new[I]
            ps.grid_ids_buffer[new_index] = ps.grid_ids[I]
            ps.object_id_buffer[new_index] = ps.object_id[I]
            ps.x_0_buffer[new_index] = ps.x_0[I]
            ps.x_buffer[new_index] = ps.x[I]
            ps.v_buffer[new_index] = ps.v[I]
            ps.acceleration_buffer[new_index] = ps.acceleration[I]
            ps.m_V_buffer[new_index] = ps.m_V[I]
            ps.m_buffer[new_index] = ps.m[I]
            ps.m_inv_buffer[new_index] = ps.m_inv[I]
            ps.density_buffer[new_index] = ps.density[I]
            ps.density0_buffer[new_index] = ps.density0[I]
            ps.pressure_buffer[new_index] = ps.pressure[I]
            ps.material_buffer[new_index] = ps.material[I]
            ps.color_buffer[new_index] = ps.color[I]
            ps.is_dynamic_buffer[new_index] = ps.is_dynamic[I]
            ps.n_buffer[new_index] = ps.n[I]

        for I in range(n):
           ps.grid_ids[I]     = ps.grid_ids_buffer[I]
           ps.object_id[I]    = ps.object_id_buffer[I]
           ps.x_0[I]          = ps.x_0_buffer[I]
           ps.x[I]            = ps.x_buffer[I]
           ps.v[I]            = ps.v_buffer[I]
           ps.acceleration[I] = ps.acceleration_buffer[I]
           ps.m_V[I]          = ps.m_V_buffer[I]
           ps.m[I]            = ps.m_buffer[I]
           ps.m_inv[I]        = ps.m_inv_buffer[I]
           ps.density[I]      = ps.density_buffer[I]
           ps.density0[I]     = ps.density0_buffer[I]
           ps.pressure[I]     = ps.pressure_buffer[I]
           ps.material[I]     = ps.material_buffer[I]
           ps.color[I]        = ps.color_buffer[I]
           ps.is_dynamic[I]   = ps.is_dynamic_buffer[I]
           ps.n[I]            = ps.n_buffer[I]

    @ti.kernel
    def narrow_phase(self, x: ti.template()):
        for p_i in range(self.ps.particle_num[None]):
            self.ps.fluid_neighbors_num[p_i] = 0
            self.ps.solid_neighbors_num[p_i] = 0
            center_cell = self.pos_to_index(x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                nbr_cell = self.clamp_cell(center_cell + offset)
                grid_index = self.flatten_grid_index(nbr_cell)
                start = 0
                if grid_index > 0:
                    start = self.ps.grid_particles_num[grid_index - 1]

                end = self.ps.grid_particles_num[grid_index]
                for p_j in range(start, end):
                    # for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                    if p_i != p_j and (x[p_i] - x[p_j]).norm() < self.ps.support_radius:
                        if self.ps.fluid_neighbors_num[p_i] < self.ps.cache_size:
                            self.ps.fluid_neighbors[p_i, self.ps.fluid_neighbors_num[p_i]] = p_j
                            self.ps.fluid_neighbors_num[p_i] += 1
                        if self.ps.material[p_i] == self.ps.material_solid and self.ps.material[p_j] == self.ps.material_solid:
                            if self.ps.solid_neighbors_num[p_i] < self.ps.cache_size:
                                self.ps.solid_neighbors[p_i, self.ps.solid_neighbors_num[p_i]] = p_j
                                self.ps.solid_neighbors_num[p_i] += 1


    def broad_phase(self):
        self.update_grid_id(self.ps)
        self.prefix_sum_executor.run(self.ps.grid_particles_num)
        self.counting_sort(self.ps)