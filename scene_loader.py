import taichi as ti
import numpy as np
import trimesh as tm
from functools import reduce
from config_builder import SimConfig
from emitter import EmitterSystem

@ti.data_oriented
class SceneLoader:
    def __init__(self, config: SimConfig):
        self.cfg = config

        self.dim = len(self.cfg.get_cfg("domainEnd"))
        self.particle_radius = float(self.cfg.get_cfg("particleRadius"))
        self.particle_diameter = 2.0 * self.particle_radius

        # Emitter timer and emitter system
        self.es = None
        self.time = 0.0

    # ===================================================================== #

    def get_scene_name(self, scene_file_path: str):
        p = scene_file_path or ""
        return p.split("/")[-1].split(".")[0] if p else "scene"


    def prepare_scene(self):
        fluid_blocks = list(self.cfg.get_fluid_blocks() or [])
        fluid_bodies = list(self.cfg.get_fluid_bodies() or [])
        rigid_blocks = list(self.cfg.get_rigid_blocks() or [])
        rigid_bodies = list(self.cfg.get_rigid_bodies() or [])

        object_ids = []
        object_map ={}

        fluid_cnt = 0
        rigid_cnt = 0
        rb_dynamic_cnt = 0

        # Process Fluid Blocks
        for fluid in fluid_blocks:
            object_ids.append(int(fluid["objectId"]))
            object_map[fluid["objectId"]] = fluid
            t = np.array(fluid.get("translation", [0, 0, 0]), dtype=np.float32)
            s = np.array(fluid.get("scale", [1, 1, 1]), dtype=np.float32)
            start = np.array(fluid["start"], dtype=np.float32) + t
            size  = (np.array(fluid["end"], dtype=np.float32) - np.array(fluid["start"], dtype=np.float32)) * s
            end   = start + size
            particle_num = self.compute_cube_particle_num(start, end)
            fluid["worldStart"] = start
            fluid["worldEnd"] = end
            fluid["particleNum"] = particle_num
            fluid_cnt += particle_num

        # Process Fluid Bodies
        for fb in fluid_bodies:
            object_ids.append(int(fb["objectId"]))
            object_map[fb["objectId"]] = fb
            voxelized_points_np = self.load_mesh(fb)
            fb["particleNum"] = voxelized_points_np.shape[0]
            fb["voxelizedPoints"] = voxelized_points_np
            fluid_cnt += voxelized_points_np.shape[0]

        # Process Rigid Blocks
        for rigid in rigid_blocks:
            object_ids.append(int(rigid["objectId"]))
            object_map[rigid["objectId"]] = rigid
            t = np.array(fluid.get("translation", [0, 0, 0]), dtype=np.float32)
            s = np.array(fluid.get("scale", [1, 1, 1]), dtype=np.float32)
            start = np.array(fluid["start"], dtype=np.float32) + t
            size  = (np.array(fluid["end"], dtype=np.float32) - np.array(fluid["start"], dtype=np.float32)) * s
            end   = start + size
            particle_num = self.compute_cube_particle_num(start, end)
            rigid["worldStart"] = start
            rigid["worldEnd"] = end
            rigid["particleNum"] = particle_num
            rigid_cnt += particle_num
            if rigid["isDynamic"]:
                rb_dynamic_cnt += particle_num

        # Process Rigid Bodies
        for rigid_body in rigid_bodies:
            object_ids.append(int(rigid_body["objectId"]))
            object_map[rigid_body["objectId"]] = rigid_body
            voxelized_points_np = self.load_mesh(rigid_body)
            rigid_body["particleNum"] = voxelized_points_np.shape[0]
            rigid_body["voxelizedPoints"] = voxelized_points_np
            rigid_cnt += voxelized_points_np.shape[0]
            if rigid_body["isDynamic"]:
                rb_dynamic_cnt += voxelized_points_np.shape[0]

        num_objects = (max(object_ids) + 1) if object_ids else 1

        # Process Emitter
        emitter_cfg_raw = self.cfg.get_emitter()
        emitter_capacity = 0
        emitter_defs = []
        emitter_reuse = False
        emitter_max_reuse_per_step = max(1, fluid_cnt // 10)

        if isinstance(emitter_cfg_raw, dict):
            emitter_capacity = int(emitter_cfg_raw.get("capacity", 0) or 0)
            emitter_reuse = bool(emitter_cfg_raw.get("reuse", False))
            emitter_max_reuse_per_step = int(emitter_cfg_raw.get("maxReusePerStep", emitter_max_reuse_per_step) or emitter_max_reuse_per_step)
            emitter_defs = list(emitter_cfg_raw.get("emitters", []) or [])
        elif isinstance(emitter_cfg_raw, list):
            emitter_defs = list(emitter_cfg_raw)

        particle_max_num = fluid_cnt + rigid_cnt + int(emitter_capacity)

        return {
            "fluid_blocks": fluid_blocks,
            "fluid_bodies": fluid_bodies,
            "rigid_blocks": rigid_blocks,
            "rigid_bodies": rigid_bodies,
            "particle_max_num": int(particle_max_num),
            "fluid_particle_num": int(fluid_cnt),
            "rigid_particle_num": int(rigid_cnt),
            "dynamic_particle_num": int(rb_dynamic_cnt),
            "num_objects": int(num_objects),
            "num_rigid_bodies": int(len(rigid_blocks) + len(rigid_bodies)),
            "object_map": object_map,
            "object_ids": object_ids,
            "emitter": {
                "capacity": int(emitter_capacity),
                "reuse": bool(emitter_reuse),
                "max_reuse_per_step": int(emitter_max_reuse_per_step),
                "defs": emitter_defs,
            }
        }


    def populate_scene(self, ps, scene_data):
        ps.allocate(
            particle_max_num=scene_data["particle_max_num"],
            num_objects=scene_data["num_objects"],
            num_rigid_bodies=scene_data["num_rigid_bodies"],
            enable_ggui=True
        )

        ps.fluid_particle_num   = int(scene_data["fluid_particle_num"])
        ps.rigid_particle_num   = int(scene_data["rigid_particle_num"])
        ps.dynamic_particle_num = int(scene_data["dynamic_particle_num"])

        ps.object_collection.clear()
        ps.object_collection.update(scene_data["object_map"])

        for fluid in scene_data["fluid_blocks"]:
            obj_id = int(fluid["objectId"])
            start, end = fluid["worldStart"], fluid["worldEnd"]
            scale = np.array(fluid["scale"])
            velocity = fluid["velocity"]
            density = fluid["density"]
            color = fluid["color"]

            self.add_cube(
                particle_system=ps,
                object_id=obj_id,
                lower_corner=start,
                cube_size=(end-start)*scale,
                velocity=velocity,
                density=density,
                color=color,
                material=1,
                is_dynamic=1
            )

        for fluid_body in scene_data["fluid_bodies"]:
            obj_id = int(fluid_body["objectId"])
            num_particles_obj = fluid_body["particleNum"]
            voxelized_points_np = fluid_body["voxelizedPoints"]
            velocity = np.array(fluid_body.get("velocity", [0.0 for _ in range(self.dim)]), dtype=np.float32)
            density = float(fluid_body.get("density", 1000.0))
            color = np.array(fluid_body.get("color", [10, 100, 200]), dtype=np.int32)
            
            ps.add_particles(obj_id,
                               num_particles_obj,
                               np.array(voxelized_points_np, dtype=np.float32), # position
                               np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                               density * np.ones(num_particles_obj, dtype=np.float32), # density
                               np.zeros(num_particles_obj, dtype=np.float32), # pressure
                               np.ones(num_particles_obj, dtype=np.int32), # material is fluid
                               np.ones(num_particles_obj, dtype=np.int32), # is_dynamic = 1
                               np.stack([color for _ in range(num_particles_obj)])) # color
        
        for rigid in scene_data["rigid_blocks"]:
            obj_id = int(rigid["objectId"])
            is_dynamic = int(bool(rigid_body.get("isDynamic")))

            start, end = rigid["worldStart"], rigid["worldEnd"]
            scale = np.array(rigid["scale"])
            velocity = rigid["velocity"]
            density = rigid["density"]
            color = rigid["color"]


            self.add_cube(
                particle_system=ps,
                object_id=obj_id,
                lower_corner=start,
                cube_size=(end-start)*scale,
                velocity=velocity,
                density=density,
                color=color,
                material=0,
                is_dynamic=is_dynamic
            )

        for rigid_body in scene_data["rigid_bodies"]:
            obj_id = int(rigid_body["objectId"])
            ps.object_id_rigid_body.add(obj_id)
            is_dynamic = int(bool(rigid_body.get("isDynamic")))
            is_toggled = bool(rigid_body.get("isToggled", False))

            num_particles_obj = rigid_body["particleNum"]
            voxelized_points_np = rigid_body["voxelizedPoints"]
            desired = np.array(rigid_body.get("velocity", [0.0 for _ in range(self.dim)]), dtype=np.float32)
            velocity = desired if is_dynamic else np.zeros(self.dim, dtype=np.float32)
            density = rigid_body["density"]
            color = np.array(rigid_body["color"], dtype=np.int32)

            if is_dynamic and is_toggled:
                ps.toggled_rigid_bodies.add(obj_id)
                ps.toggled_dynamic_velocity[obj_id] = desired
            
            ps.add_particles(obj_id,
                                num_particles_obj,
                                np.array(voxelized_points_np, dtype=np.float32), # position
                                np.stack([velocity for _ in range(num_particles_obj)]), # velocity
                                density * np.ones(num_particles_obj, dtype=np.float32), # density
                                np.zeros(num_particles_obj, dtype=np.float32), # pressure
                                np.zeros(num_particles_obj, dtype=np.int32), # material is solid
                                is_dynamic * np.ones(num_particles_obj, dtype=np.int32),
                                np.stack([color for _ in range(num_particles_obj)])) # color
        
        self._setup_emitter_system(ps, scene_data["emitter"])


    def load_mesh(self, obj, a=1.0):
        obj_id = obj["objectId"]
        mesh = tm.load(obj["geometryFile"])
        mesh.apply_scale(obj["scale"])
        offset = np.array(obj["translation"])

        angle = obj["rotationAngle"] / 360 * 2 * 3.1415926
        direction = obj["rotationAxis"]
        rot_matrix = tm.transformations.rotation_matrix(angle, direction, mesh.vertices.mean(axis=0))
        mesh.apply_transform(rot_matrix)
        mesh.vertices += offset
        
        # Backup the original mesh for exporting obj
        mesh_backup = mesh.copy()
        obj["mesh"] = mesh_backup
        obj["restPosition"] = mesh_backup.vertices
        obj["restCenterOfMass"] = mesh_backup.vertices.mean(axis=0)
        is_success = tm.repair.fill_holes(mesh)
            # print("Is the mesh successfully repaired? ", is_success)

        voxelized_mesh = mesh.voxelized(pitch=a * self.particle_diameter)
        voxelized_mesh = mesh.voxelized(pitch=a * self.particle_diameter).fill()
        # voxelized_mesh = mesh.voxelized(pitch=self.particle_diameter).hollow()
        # voxelized_mesh.show()
        voxelized_points_np = voxelized_mesh.points
        print(f"Mesh object {obj_id} loaded. Particle count: {voxelized_points_np.shape[0]}")
        
        return voxelized_points_np


    def compute_cube_particle_num(self, start, end):
        num_dim = [np.arange(start[i], end[i], self.particle_diameter) for i in range(self.dim)]
        n = 1
        for arr in num_dim:
            n *= len(arr)
        return int(n)


    def add_cube(self,
                particle_system,
                object_id,
                lower_corner,
                cube_size,
                material,
                is_dynamic,
                color=(0,0,0),
                density=None,
                pressure=None,
                velocity=None):
        ps = particle_system
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
        ps.add_particles(object_id, num_new_particles, new_positions, velocity_arr, density_arr, pressure_arr, material_arr, is_dynamic_arr, color_arr)
    

    #==== Emitter system setup ====
    def reset_emitter_system(self):
        if getattr(self, "es", None):
            self.es.reset()
        self.time = 0.0

    def step_emitter_system(self, dt: float):
        if getattr(self, "es", None):
            em, ru = self.es.step(self.time, float(dt))
            self.time += float(dt)
            return em, ru
        return 0, 0


    def _build_rotation_from_direction(self, direction_np: np.ndarray):
        d = np.array(direction_np, dtype=np.float32)
        n = np.linalg.norm(d) + 1e-12
        ex = d / n
        # Choose a stable reference up and project it onto the plane orthogonal to ex
        ref_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(ex, ref_up)) > 0.99:
            ref_up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # ey: as aligned with world-up as possible but orthogonal to ex
        ey = ref_up - np.dot(ref_up, ex) * ex
        ey = ey / (np.linalg.norm(ey) + 1e-12)
        # ez completes right-handed frame
        ez = np.cross(ex, ey)
        ez = ez / (np.linalg.norm(ez) + 1e-12)
        # Columns: emit_dir(X), axis_h(Y), axis_w(Z)
        R = np.stack([ex, ey, ez], axis=1).astype(np.float32)
        return R


    def _setup_emitter_system(self, ps, emitter_info):
        defs = list(emitter_info.get("defs", []))
        if len(defs) == 0:
            self.es = None
            setattr(ps, "emitter_system", None)
            return

        max_reuse = int(emitter_info.get("max_reuse_per_step", 1000) or 1000)
        reuse = bool(emitter_info.get("reuse", False))

        self.es = EmitterSystem(ps, max_reuse_per_step=max_reuse)

        if reuse:
            box_min = ps.domain_start.astype(np.float32) + ps.padding
            box_max = (ps.domain_start + ps.domain_size).astype(np.float32) - ps.padding
            self.es.enable_reuse_particles(box_min=box_min, box_max=box_max)

        for e in defs:
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
            if oid not in ps.object_collection:
                ps.object_collection[oid] = {"objectId": oid}
            self.es.add_emitter(width=width,
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

        setattr(ps, "emitter_system", self.es)
