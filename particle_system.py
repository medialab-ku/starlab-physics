import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from config_builder import SimConfig


@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        self.domain_start = np.array(self.cfg.get_cfg("domainStart"), dtype=np.float32)
        self.domain_end   = np.array(self.cfg.get_cfg("domainEnd"), dtype=np.float32)
        self.domain_size  = self.domain_end - self.domain_start
        self.dim = len(self.domain_size)

        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim
        self.padding = self.support_radius

        self.x_vis_buffer = None

        # objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()
        self.num_objects = 1
        self.num_rigid_bodies = 0
        self.fluid_particle_num = 0
        self.rigid_particle_num = 0
        self.dynamic_particle_num = 0
        self.particle_max_num = 0
        self.particle_num = ti.field(int, shape=())

        # Toggle-able rigid bodies
        self.toggled_rigid_bodies = set()
        self.toggled_dynamic_velocity = {}
        self.toggled_activated = set()
        self.toggled_ids_sorted = []
        self.toggled_index = 0


    def allocate(self, particle_max_num, num_objects, num_rigid_bodies, enable_ggui:bool):
        self.particle_max_num = int(particle_max_num)
        self.num_objects = int(max(num_objects, 1))
        self.num_rigid_bodies = int(num_rigid_bodies)
        self.object_particle_num = ti.field(dtype=int, shape=self.num_objects)

        # Rigid body properties
        print(f"Number of rigid bodies: {self.num_rigid_bodies}")
        if self.num_rigid_bodies > 0:
            self.rigid_rest_cm = ti.Vector.field(self.dim, dtype=float, shape=self.num_objects)
            self.mass_rb       = ti.field(dtype=float, shape=self.num_objects)
            self.cm            = ti.Vector.field(self.dim, dtype=float, shape=self.num_objects)
            self.R             = ti.Matrix.field(self.dim, self.dim, dtype=float, shape=self.num_objects)
            self.body_mass     = ti.field(dtype=float, shape=self.num_objects)
            self.v_cm_rb       = ti.Vector.field(self.dim, dtype=float, shape=self.num_objects)
            self.omega_rb      = ti.Vector.field(self.dim, dtype=float, shape=self.num_objects)
            
        # Particle related properties
        self.object_id = ti.field(dtype=int, shape=self.particle_max_num)
        self.x = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_old = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0 = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v_adv = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v_old = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.y = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V = ti.field(dtype=float, shape=self.particle_max_num)
        self.m = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_inv = ti.field(dtype=float, shape=self.particle_max_num)
        self.n = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.density  = ti.field(dtype=float, shape=self.particle_max_num)
        self.density0 = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure = ti.field(dtype=float, shape=self.particle_max_num)
        self.divergence = ti.field(dtype=float, shape=self.particle_max_num)
        self.material = ti.field(dtype=int, shape=self.particle_max_num)
        self.color = ti.Vector.field(4, dtype=int, shape=self.particle_max_num) # RGBA
        self.is_dynamic = ti.field(dtype=int, shape=self.particle_max_num)

        self.cur2ori = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.ori2cur = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)

        # neighbor lists
        self.cache_size = 50 
        self.particle_neighbors_num = ti.field(dtype=int, shape=self.particle_max_num)
        self.solid_neighbors_num    = ti.field(dtype=int, shape=self.particle_max_num)
        self.particle_neighbors        = ti.field(dtype=int, shape=(self.particle_max_num, self.cache_size))
        self.solid_neighbors        = ti.field(dtype=int, shape=(self.particle_max_num, self.cache_size))
        self.fluid_neighbors_values = ti.Vector.field(n=3, dtype=float, shape=(self.particle_max_num, self.cache_size))
        self.fluid_neighbors_JtJ    = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.particle_max_num, self.cache_size))
        self.fluid_neighbors_JtJ_ii = ti.Matrix.field(n=3, m=3, dtype=float, shape= self.particle_max_num)


        # Buffer for sort
        self.object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.x_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_inv_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_buffer  = ti.field(dtype=float, shape=self.particle_max_num)
        self.density0_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.color_buffer = ti.Vector.field(4, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.n_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)

        self.cur2ori_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)

        # Special properties
        method = int(self.cfg.get_cfg("simulationMethod") or 0)
        if method == 4:
            self.dfsph_factor = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv  = ti.field(dtype=float, shape=self.particle_max_num)
            self.dfsph_factor_buffer = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv_buffer  = ti.field(dtype=float, shape=self.particle_max_num)

        if enable_ggui and self.GGUI:
            self.x_vis_buffer     = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(4, dtype=float, shape=self.particle_max_num)
            self.color_heat_map   = ti.Vector.field(4, dtype=float, shape=self.particle_max_num)


    # Helper and util methods
    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.object_id[p] = obj_id
        self.x[p] = x
        self.x_old[p] = x
        self.x_0[p] = x
        self.v[p] = v
        self.v_adv[p] = v
        self.v_old[p] = v
        self.acceleration[p] = ti.Vector.zero(float, self.dim)
        self.n[p] = ti.Vector.zero(float, self.dim)
        self.density0[p] = self.density[p] = density
        self.m_V[p] = self.m_V0
        self.m[p] = self.m_V0 * self.density0[p]

        self.pressure[p] = pressure
        self.material[p] = material
        self.is_dynamic[p] = is_dynamic

        if is_dynamic:
            self.m_inv[p] = 1.0 / self.m[p]
        else:
            self.m_inv[p] = 0.0
            self.m[p] *= 10.0
        self.color[p] = color
    

    def add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()
                      ):
        
        self._add_particles(object_id,
                      new_particles_num,
                      new_particles_positions,
                      new_particles_velocity,
                      new_particle_density,
                      new_particle_pressure,
                      new_particles_material,
                      new_particles_is_dynamic,
                      new_particles_color
                      )


    @ti.kernel
    def _add_particles(self,
                      object_id: int,
                      new_particles_num: int,
                      new_particles_positions: ti.types.ndarray(),
                      new_particles_velocity: ti.types.ndarray(),
                      new_particle_density: ti.types.ndarray(),
                      new_particle_pressure: ti.types.ndarray(),
                      new_particles_material: ti.types.ndarray(),
                      new_particles_is_dynamic: ti.types.ndarray(),
                      new_particles_color: ti.types.ndarray()):
        for p in range(self.particle_num[None], self.particle_num[None] + new_particles_num):
            v = ti.Vector.zero(float, self.dim)
            x = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                v[d] = new_particles_velocity[p - self.particle_num[None], d]
                x[d] = new_particles_positions[p - self.particle_num[None], d]
            self.add_particle(p, object_id, x, v,
                              new_particle_density[p - self.particle_num[None]],
                              new_particle_pressure[p - self.particle_num[None]],
                              new_particles_material[p - self.particle_num[None]],
                              new_particles_is_dynamic[p - self.particle_num[None]],
                              ti.Vector([new_particles_color[p - self.particle_num[None], 0],
                                         new_particles_color[p - self.particle_num[None], 1],
                                         new_particles_color[p - self.particle_num[None], 2],
                                         255])
                              )
        self.particle_num[None] += new_particles_num
    

    def set_emitted_on_indices(self, count, idxs_np, P_np, V_np, object_id, density, color_np):
        r, g, b = int(color_np[0]), int(color_np[1]), int(color_np[2])
        self._set_emitted_on_indices(count,
                                     idxs_np.astype(np.int32),
                                     P_np.astype(np.float32),
                                     V_np.astype(np.float32),
                                     int(object_id),
                                     float(density),
                                     r, g, b)
        

    @ti.kernel
    def _set_emitted_on_indices(self,
                                count: int,
                                idxs: ti.types.ndarray(),
                                positions: ti.types.ndarray(),
                                velocities: ti.types.ndarray(),
                                object_id: int,
                                density: float,
                                color_r: int, color_g: int, color_b: int):
        for k in range(count):
            p = idxs[k]
            x = ti.Vector.zero(float, self.dim)
            v = ti.Vector.zero(float, self.dim)
            for d in ti.static(range(self.dim)):
                x[d] = positions[k, d]
                v[d] = velocities[k, d]
            self.object_id[p] = object_id
            self.x[p] = x
            self.x_old[p] = x
            self.x_0[p] = x
            self.v[p] = v
            self.v_adv[p] = v
            self.v_old[p] = v
            self.acceleration[p] = ti.Vector.zero(float, self.dim)
            self.n[p] = ti.Vector.zero(float, self.dim)
            self.density0[p] = self.density[p] = density
            self.pressure[p] = 0.0
            self.m_V[p] = self.m_V0
            self.m[p] = self.m_V0 * self.density0[p]
            self.m_inv[p] = 1.0 / (self.m[p] + 1e-12)
            self.material[p] = self.material_fluid
            self.is_dynamic[p] = 1
            self.color[p] = ti.Vector([color_r, color_g, color_b, 255])


    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])


    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]


    @ti.kernel
    def initialize_rigid_mass(self):
        self.mass_rb.fill(0.0)
        for p_i in ti.grouped(self.x):
            # Condition for boundary particles
            if self.is_dynamic_rigid_body(p_i):
                
                object_id = self.object_id[p_i]
                self.mass_rb[object_id] += self.m[p_i]


    @ti.kernel
    def initialize_object_particle_num(self):
        # reset counts
        for i in ti.grouped(self.object_particle_num):
            self.object_particle_num[i] = 0
        # accumulate counts per object id over active particles
        for p_i in range(self.particle_num[None]):
            obj_id = self.object_id[p_i]
            ti.atomic_add(self.object_particle_num[obj_id], 1)


    @ti.kernel
    def initialize_boundary_neighbors(self):
        for p_i in ti.grouped(self.x):
            sum_Wij = 0.0
            # Condition for boundary particles
            if self.material[p_i] == self.material_solid:

                for j in range(self.particle_neighbors_num[p_i]):
                    p_j = self.particle_neighbors[p_i, j]
                    if self.material[p_j] != self.material_solid:
                        continue

                    if self.object_id[p_j] != self.object_id[p_i]:
                        continue

                    sum_Wij += self.solver.W((self.x[p_i] - self.x[p_j]).norm())

                if sum_Wij > 1e-12:
                    self.m[p_i] = 1.5*self.density0[p_i] / sum_Wij
                    self.m_V[p_i] = self.m[p_i] / self.density0[p_i]
                    # Keep inverse mass consistent (static solids keep 0 inv mass)
                    if self.is_dynamic[p_i]:
                        self.m_inv[p_i] = 1.0 / (self.m[p_i] + 1e-12)


    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for d in ti.static(range(self.dim)):
                np_arr[i, d] = src_arr[i][d]


    def dump(self, obj_id):
        N = int(self.particle_num[None])
        np_object_id = self.object_id.to_numpy()[:N]
        mask = (np_object_id == obj_id)
        np_x = self.x.to_numpy()[:N][mask]
        np_v = self.v.to_numpy()[:N][mask]

        return {
            'position': np_x,
            'velocity': np_v
        }