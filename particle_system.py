import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from config_builder import SimConfig
from WCSPH import WCSPHSolver
from DFSPH import DFSPHSolver
from PBF2 import PBF2Solver
from IISPH import IISPHSolver
from scan_single_buffer import parallel_prefix_sum_inclusive_inplace
from emitter import EmitterSystem


@ti.data_oriented
class ParticleSystem:
    def __init__(self, config: SimConfig, GGUI=False):
        self.cfg = config
        self.GGUI = GGUI

        self.domain_start = np.array([0.0, 0.0, 0.0])
        self.domain_start = np.array(self.cfg.get_cfg("domainStart"))

        self.domain_end = np.array([1.0, 1.0, 1.0])
        self.domian_end = np.array(self.cfg.get_cfg("domainEnd"))
        
        self.domain_size = self.domian_end - self.domain_start

        self.dim = len(self.domain_size)
        assert self.dim > 1
        # Simulation method
        self.simulation_method = self.cfg.get_cfg("simulationMethod")

        # Material
        self.material_solid = 0
        self.material_fluid = 1

        self.particle_radius = 0.01  # particle radius
        self.particle_radius = self.cfg.get_cfg("particleRadius")

        self.particle_diameter = 2 * self.particle_radius
        self.support_radius = self.particle_radius * 4.0  # support radius
        self.m_V0 = 0.8 * self.particle_diameter ** self.dim

        self.particle_num = ti.field(int, shape=())

        # Grid related properties
        self.grid_size = self.support_radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size

        # All objects id and its particle num
        self.object_collection = dict()
        self.object_id_rigid_body = set()

        # Emitter timer and emitter system
        self.emitter_system = None
        self.time = 0.0

        #========== Compute number of particles ==========#
        #### Process Fluid Blocks ####
        fluid_blocks = self.cfg.get_fluid_blocks()
        fluid_particle_num = 0
        for fluid in fluid_blocks:
            particle_num = self.compute_cube_particle_num(fluid["start"], fluid["end"])
            fluid["particleNum"] = particle_num
            self.object_collection[fluid["objectId"]] = fluid
            fluid_particle_num += particle_num

        #### Process Fluid Bodies ####
        fluid_bodies = self.cfg.get_fluid_bodies()
        for fb in fluid_bodies:
            if fb.get("meshlab", False):
                voxelized_points_np = self.load_rigid_body_meshlab(fb)
            else:
                voxelized_points_np = self.load_rigid_body(fb)
            fb["particleNum"] = voxelized_points_np.shape[0]
            fb["voxelizedPoints"] = voxelized_points_np
            self.object_collection[fb["objectId"]] = fb
            fluid_particle_num += voxelized_points_np.shape[0]

        #### Process Rigid Blocks ####
        rigid_blocks = self.cfg.get_rigid_blocks()
        rigid_particle_num = 0
        rigid_dynamic_particle_num = 0
        for rigid in rigid_blocks:
            particle_num = self.compute_cube_particle_num(rigid["start"], rigid["end"])
            rigid["particleNum"] = particle_num
            self.object_collection[rigid["objectId"]] = rigid
            rigid_particle_num += particle_num
            if rigid["isDynamic"]:
                rigid_dynamic_particle_num += particle_num
        
        #### Process Rigid Bodies ####
        rigid_bodies = self.cfg.get_rigid_bodies()
        for rigid_body in rigid_bodies:
            voxelized_points_np = self.load_rigid_body(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            self.object_collection[rigid_body["objectId"]] = rigid_body
            rigid_particle_num += voxelized_points_np.shape[0]
            if rigid_body["isDynamic"]:
                rigid_dynamic_particle_num += voxelized_points_np.shape[0]

        object_ids = list(self.object_collection.keys())
        self.num_objects = (max(object_ids) + 1) if len(object_ids) > 0 else 1
        
        self.fluid_particle_num = fluid_particle_num
        self.solid_particle_num = rigid_particle_num
        self.dynamic_particle_num = fluid_particle_num + rigid_dynamic_particle_num
        self.particle_max_num = fluid_particle_num + rigid_particle_num
        self.num_rigid_bodies = len(rigid_blocks)+len(rigid_bodies)
        # Use max object id + 1 to avoid out-of-bounds when indexing by object_id
        self.object_particle_num = ti.field(dtype=int, shape=self.num_objects)

        #========== Particle Emitter ==========#
        emitter_cfg_raw = self.cfg.get_emitter()
        self.emitter_capacity = 0
        self.emitter_max_reuse_per_step = self.fluid_particle_num // 10
        self.emitter_defs = []
        if emitter_cfg_raw:
            if isinstance(emitter_cfg_raw, dict):
                self.emitter_capacity = int(emitter_cfg_raw.get("capacity", 0) or 0)
                self.emitter_reuse = bool(emitter_cfg_raw.get("reuse", False))
                self.emitter_max_reuse_per_step = int(emitter_cfg_raw.get("maxReusePerStep", 50000) or 50000)
                self.emitter_defs = list(emitter_cfg_raw.get("emitters", []) or [])
            elif isinstance(emitter_cfg_raw, list):
                self.emitter_defs = list(emitter_cfg_raw)
        if self.emitter_capacity > 0:
            self.particle_max_num += self.emitter_capacity

        #========== Allocate memory ==========#
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
            
            
        # Particle num of each grid
        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))

        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

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

        self.cache_size = 50 
        self.fluid_neighbors_num    = ti.field(dtype=int, shape=self.particle_max_num)
        self.solid_neighbors_num    = ti.field(dtype=int, shape=self.particle_max_num)
        self.fluid_neighbors        = ti.field(dtype=int, shape=(self.particle_max_num, self.cache_size))
        self.solid_neighbors        = ti.field(dtype=int, shape=(self.particle_max_num, self.cache_size))
        self.fluid_neighbors_values = ti.Vector.field(n=3, dtype=float, shape=(self.particle_max_num, self.cache_size))

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv = ti.field(dtype=float, shape=self.particle_max_num)

        # Buffer for sort
        self.object_id_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.x_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.x_0_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.v_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.acceleration_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
        self.m_V_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.m_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.density_buffer  = ti.field(dtype=float, shape=self.particle_max_num)
        self.density0_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.pressure_buffer = ti.field(dtype=float, shape=self.particle_max_num)
        self.material_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.color_buffer = ti.Vector.field(4, dtype=int, shape=self.particle_max_num)
        self.is_dynamic_buffer = ti.field(dtype=int, shape=self.particle_max_num)
        self.n_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)

        if self.cfg.get_cfg("simulationMethod") == 4:
            self.dfsph_factor_buffer = ti.field(dtype=float, shape=self.particle_max_num)
            self.density_adv_buffer = ti.field(dtype=float, shape=self.particle_max_num)

        # Grid id for each particle
        self.grid_ids = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_buffer = ti.field(int, shape=self.particle_max_num)
        self.grid_ids_new = ti.field(int, shape=self.particle_max_num)

        self.x_vis_buffer = None
        if self.GGUI:
            self.x_vis_buffer = ti.Vector.field(self.dim, dtype=float, shape=self.particle_max_num)
            self.color_vis_buffer = ti.Vector.field(4, dtype=float, shape=self.particle_max_num)
            self.color_heat_map = ti.Vector.field(4, dtype=float, shape=self.particle_max_num)


        #========== Initialize particles ==========#

        # Fluid block
        for fluid in fluid_blocks:
            obj_id = fluid["objectId"]
            offset = np.array(fluid["translation"])
            start = np.array(fluid["start"]) + offset
            end = np.array(fluid["end"]) + offset
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]

            # print(density)
            # print(color)

            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=1, # enforce fluid dynamic
                          color=color,
                          material=1) # 1 indicates fluid

        # Fluid bodies
        for fluid_body in fluid_bodies:
            obj_id = fluid_body["objectId"]
            num_particles_obj = fluid_body["particleNum"]
            voxelized_points_np = fluid_body["voxelizedPoints"]
            velocity = np.array(fluid_body.get("velocity", [0.0 for _ in range(self.dim)]), dtype=np.float32)
            density = float(fluid_body.get("density", 1000.0))
            color = np.array(fluid_body.get("color", [10, 100, 200]), dtype=np.int32)
            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32), # position
                               np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32), # density
                               np.zeros(num_particles_obj, dtype=np.float32), # pressure
                               np.array([self.material_fluid for _ in range(num_particles_obj)], dtype=np.int32), # material is fluid
                               np.ones(num_particles_obj, dtype=np.int32), # is_dynamic = 1
                               np.stack([color for _ in range(num_particles_obj)])) # color

        # Rigid block
        for rigid in rigid_blocks:
            obj_id = rigid["objectId"]
            offset = np.array(rigid["translation"])
            start = np.array(rigid["start"]) + offset
            end = np.array(rigid["end"]) + offset
            scale = np.array(rigid["scale"])
            velocity = rigid["velocity"]
            density = rigid["density"]
            color = rigid["color"]
            is_dynamic = rigid["isDynamic"]
            self.add_cube(object_id=obj_id,
                          lower_corner=start,
                          cube_size=(end-start)*scale,
                          velocity=velocity,
                          density=density, 
                          is_dynamic=is_dynamic,
                          color=color,
                          material=0) # 0 indicates solid


        # Rigid bodies
        for rigid_body in rigid_bodies:
            obj_id = rigid_body["objectId"]
            self.object_id_rigid_body.add(obj_id)
            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            is_dynamic = rigid_body["isDynamic"]
            if is_dynamic:
                velocity = np.array(rigid_body["velocity"], dtype=np.float32)
            else:
                velocity = np.array([0.0 for _ in range(self.dim)], dtype=np.float32)
            density = rigid_body["density"]
            color = np.array(rigid_body["color"], dtype=np.int32)
            self.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32), # position
                               np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32), # density
                               np.zeros(num_particles_obj, dtype=np.float32), # pressure
                               np.array([0 for _ in range(num_particles_obj)], dtype=np.int32), # material is solid
                               is_dynamic * np.ones(num_particles_obj, dtype=np.int32), # is_dynamic
                               np.stack([color for _ in range(num_particles_obj)])) # color
            
        self._setup_emitter_system()


    def build_solver(self):
        solver_type = self.cfg.get_cfg("simulationMethod")
        if solver_type == 0:
            self.solver = WCSPHSolver(self)
            return WCSPHSolver(self)
        elif solver_type == 2:
            self.solver = PBF2Solver(self)
            return PBF2Solver(self)
        elif solver_type == 3:
            self.solver = IISPHSolver(self)
            return IISPHSolver(self)
        elif solver_type == 4:
            self.solver = DFSPHSolver(self)
            return DFSPHSolver(self)

        else:
            raise NotImplementedError(f"Solver type {solver_type} has not been implemented.")

    @ti.func
    def add_particle(self, p, obj_id, x, v, density, pressure, material, is_dynamic, color):
        self.object_id[p] = obj_id
        self.x[p] = x
        self.x_0[p] = x
        self.v[p] = v
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

    def set_emitted_on_indices(self, count, idxs_np, P_np, V_np, object_id, density, color_np):
        r, g, b = int(color_np[0]), int(color_np[1]), int(color_np[2])
        self._set_emitted_on_indices(count,
                                     idxs_np.astype(np.int32),
                                     P_np.astype(np.float32),
                                     V_np.astype(np.float32),
                                     int(object_id),
                                     float(density),
                                     r, g, b)
        

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
    

    @ti.func
    def is_static_rigid_body(self, p):
        return self.material[p] == self.material_solid and (not self.is_dynamic[p])


    @ti.func
    def is_dynamic_rigid_body(self, p):
        return self.material[p] == self.material_solid and self.is_dynamic[p]
    
    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0
        for p in range(self.particle_num[None]):
            grid_index = self.get_flatten_grid_index(self.x[p])
            self.grid_ids[p] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]
    

    @ti.kernel
    def counting_sort(self):
        n = self.particle_num[None]

        for i in range(n):
            I = n - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I] - 1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

        for I in range(n):
            new_index = self.grid_ids_new[I]
            self.grid_ids_buffer[new_index] = self.grid_ids[I]
            self.object_id_buffer[new_index] = self.object_id[I]
            self.x_0_buffer[new_index] = self.x_0[I]
            self.x_buffer[new_index] = self.x[I]
            self.v_buffer[new_index] = self.v[I]
            self.acceleration_buffer[new_index] = self.acceleration[I]
            self.m_V_buffer[new_index] = self.m_V[I]
            self.m_buffer[new_index] = self.m[I]
            self.density_buffer[new_index] = self.density[I]
            self.density0_buffer[new_index] = self.density0[I]
            self.pressure_buffer[new_index] = self.pressure[I]
            self.material_buffer[new_index] = self.material[I]
            self.color_buffer[new_index] = self.color[I]
            self.is_dynamic_buffer[new_index] = self.is_dynamic[I]
            self.n_buffer[new_index] = self.n[I]
            if ti.static(self.simulation_method == 4):
                self.dfsph_factor_buffer[new_index] = self.dfsph_factor[I]
                self.density_adv_buffer[new_index] = self.density_adv[I]

        for I in range(n):
            self.grid_ids[I] = self.grid_ids_buffer[I]
            self.object_id[I] = self.object_id_buffer[I]
            self.x_0[I] = self.x_0_buffer[I]
            self.x[I] = self.x_buffer[I]
            self.v[I] = self.v_buffer[I]
            self.acceleration[I] = self.acceleration_buffer[I]
            self.m_V[I] = self.m_V_buffer[I]
            self.m[I] = self.m_buffer[I]
            self.density[I] = self.density_buffer[I]
            self.density0[I] = self.density0_buffer[I]
            self.pressure[I] = self.pressure_buffer[I]
            self.material[I] = self.material_buffer[I]
            self.color[I] = self.color_buffer[I]
            self.is_dynamic[I] = self.is_dynamic_buffer[I]
            self.n[I] = self.n_buffer[I]
            if ti.static(self.simulation_method == 4):
                self.dfsph_factor[I] = self.dfsph_factor_buffer[I]
                self.density_adv[I] = self.density_adv_buffer[I]
    

    def initialize_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()


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

                for j in range(self.fluid_neighbors_num[p_i]):
                    p_j = self.fluid_neighbors[p_i, j]
                    if self.material[p_j] != self.material_solid:
                        continue

                    if self.object_id[p_j] != self.object_id[p_i]:
                        continue

                    sum_Wij += self.solver.Wij((self.x[p_i] - self.x[p_j]).norm())

                if sum_Wij > 1e-12:
                    self.m[p_i] = 0.7 * self.density0[p_i] / sum_Wij

    @ti.kernel
    def search_neighbours(self, x: ti.template()):
        for p_i in range(self.particle_num[None]):
            self.fluid_neighbors_num[p_i] = 0
            self.solid_neighbors_num[p_i] = 0
            center_cell = self.pos_to_index(x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
                nbr_cell = self.clamp_cell(center_cell + offset)
                grid_index = self.flatten_grid_index(nbr_cell)
                for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                    if p_i != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.support_radius:
                        if self.fluid_neighbors_num[p_i] < self.cache_size:
                            self.fluid_neighbors[p_i, self.fluid_neighbors_num[p_i]] = p_j 
                            self.fluid_neighbors_num[p_i] += 1
                        if self.material[p_i] == self.material_solid and self.material[p_j] == self.material_solid:
                            if self.solid_neighbors_num[p_i] < self.cache_size:
                                self.solid_neighbors[p_i, self.solid_neighbors_num[p_i]] = p_j 
                                self.solid_neighbors_num[p_i] += 1


    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template(), ret: ti.template()):
        center_cell = self.pos_to_index(self.x[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.dim)):
            nbr_cell = self.clamp_cell(center_cell + offset)
            # grid_index = self.flatten_grid_index(center_cell + offset)
            grid_index = self.flatten_grid_index(nbr_cell)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                if p_i[0] != p_j and (self.x[p_i] - self.x[p_j]).norm() < self.support_radius:
                    task(p_i, p_j, ret)


    @ti.kernel
    def copy_to_numpy(self, np_arr: ti.types.ndarray(), src_arr: ti.template()):
        for i in range(self.particle_num[None]):
            for d in ti.static(range(self.dim)):
                np_arr[i, d] = src_arr[i][d]
    

    def copy_to_vis_buffer(self, invisible_objects=[]):
        # Always clear buffers to avoid rendering inactive/emitted stale particles
        self.x_vis_buffer.fill(1000.0)  # move non-active far away
        self.color_vis_buffer.fill(0.0)
        for obj_id in self.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)


    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.GGUI
        # Only copy active particles
        for i in range(self.particle_num[None]):
            if self.object_id[i] == obj_id:
                self.x_vis_buffer[i] = self.x[i]
                self.color_vis_buffer[i] = self.color[i] / 255.0


    def dump(self, obj_id):
        np_object_id = self.object_id.to_numpy()
        mask = (np_object_id == obj_id).nonzero()
        np_x = self.x.to_numpy()[mask]
        np_v = self.v.to_numpy()[mask]

        return {
            'position': np_x,
            'velocity': np_v
        }
    
    def load_rigid_body(self, rigid_body):
        obj_id = rigid_body["objectId"]
        mesh = tm.load(rigid_body["geometryFile"])
        mesh.apply_scale(rigid_body["scale"])
        offset = np.array(rigid_body["translation"])

        angle = rigid_body["rotationAngle"] / 360 * 2 * 3.1415926
        direction = rigid_body["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
        
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        rigid_body["mesh"] = mesh_backup
        rigid_body["restPosition"] = mesh_backup.vertices
        rigid_body["restCenterOfMass"] = mesh_backup.vertices.mean(axis=0)
        is_success = tm.repair.fill_holes(mesh)
            # print("Is the mesh successfully repaired? ", is_success)

        a = 1.0 
        voxelized_mesh = mesh.voxelized(pitch=a * self.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=a * self.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        # voxelized_mesh.show()
        voxelized_points_np = voxelized_mesh.points
        # print(f"rigid body {obj_id} num: {voxelized_points_np.shape[0]}")
        
        return voxelized_points_np


    def compute_cube_particle_num(self, start, end):
        num_dim = []
        for i in range(self.dim):
            num_dim.append(
                np.arange(start[i], end[i], self.particle_diameter))
        return reduce(lambda x, y: x * y,
                                   [len(n) for n in num_dim])

    def add_cube(self,
                 object_id,
                 lower_corner,
                 cube_size,
                 material,
                 is_dynamic,
                 color=(0,0,0),
                 density=None,
                 pressure=None,
                 velocity=None):

        num_dim = []
        for i in range(self.dim):
            num_dim.append(np.arange(lower_corner[i], lower_corner[i] + cube_size[i], self.particle_diameter))
        num_new_particles = reduce(lambda x, y: x * y, [len(n) for n in num_dim])
        # print('particle num ', num_new_particles)

        new_positions = np.array(np.meshgrid(*num_dim,
                                             sparse=False,
                                             indexing='ij'),
                                 dtype=np.float32)
        new_positions = new_positions.reshape(-1,
                                              reduce(lambda x, y: x * y, list(new_positions.shape[1:]))).transpose()
        # print("new position shape ", new_positions.shape)
        if velocity is None:
            velocity_arr = np.full_like(new_positions, 0, dtype=np.float32)
        else:
            velocity_arr = np.array([velocity for _ in range(num_new_particles)], dtype=np.float32)

        material_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), material)
        is_dynamic_arr = np.full_like(np.zeros(num_new_particles, dtype=np.int32), is_dynamic)
        color_arr = np.stack([np.full_like(np.zeros(num_new_particles, dtype=np.int32), c) for c in color], axis=1)
        density_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), density if density is not None else 1000.)
        pressure_arr = np.full_like(np.zeros(num_new_particles, dtype=np.float32), pressure if pressure is not None else 0.)
        self.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)

    #==== Emitter system setup ====
    def _build_rotation_from_direction(self, direction_np: np.ndarray):
        d = np.array(direction_np, dtype=np.float32)
        n = np.linalg.norm(d) + 1e-12
        ex = d / n
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(ex, up)) > 0.95:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        ez = np.cross(ex, up); ez = ez / (np.linalg.norm(ez) + 1e-12)
        ey = np.cross(ez, ex); ey = ey / (np.linalg.norm(ey) + 1e-12)
        # Columns: emit_dir(X), axis_h(Y), axis_w(Z)
        R = np.stack([ex, ey, ez], axis=1).astype(np.float32)
        return R

    def _setup_emitter_system(self):
        if len(self.emitter_defs) == 0:
            return
        self.emitter_system = EmitterSystem(self, max_reuse_per_step=self.emitter_max_reuse_per_step)
        if self.emitter_reuse:
            box_min = self.domain_start.astype(np.float32) + self.padding
            box_max = (self.domain_start + self.domain_size).astype(np.float32) - self.padding
            self.emitter_system.enable_reuse_particles(box_min=box_min, box_max=box_max)
        for e in self.emitter_defs:
            typ = e.get("type", "square")
            typ_i = 0 if (isinstance(typ, str) and typ.lower() == "square") else (1 if (isinstance(typ, str) and typ.lower() == "circle") else int(typ))
            width = int(e.get("width", 1))
            height = int(e.get("height", max(1, width)))
            pos = np.array(e.get("position", [0.0, 0.0, 0.0]), dtype=np.float32)
            direction = np.array(e.get("direction", [1.0, 0.0, 0.0]), dtype=np.float32)
            R = self._build_rotation_from_direction(direction)
            vel = float(e.get("velocity", 0.0))
            jitter = float(e.get("jitter", 0.0))
            spread = float(e.get("spread", 0.0))
            st = float(e.get("startTime", 0.0))
            et = float(e.get("endTime", 1e9))
            oid = int(e.get("objectId", 0))
            density = float(e.get("density", 1000.0))
            color = np.array(e.get("color", [0, 150, 255]), dtype=np.int32)
            # Ensure emitter object id is included for visualization copy
            if oid not in self.object_collection:
                self.object_collection[oid] = {"objectId": oid}
            self.emitter_system.add_emitter(width=width,
                                            height=height,
                                            pos=pos,
                                            rotation=R,
                                            velocity=vel,
                                            type=typ_i,
                                            start_time=st,
                                            end_time=et,
                                            object_id=oid,
                                            density=density,
                                            color=tuple(color.tolist()),
                                            jitter=jitter,
                                            spread_deg=spread)