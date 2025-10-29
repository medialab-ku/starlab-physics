import taichi as ti
import numpy as np
from dataclasses import dataclass
from enum import Enum
from matplotlib.colors import Normalize, LinearSegmentedColormap

class ColorMode(str, Enum):
    heatmap = "heatmap"
    original = "original"

class HeatmapField(str, Enum):
    velocity = "velocity"
    divergence = "divergence"
    density = "density"


@dataclass
class VisualizationSettings:
    color_mode: ColorMode = ColorMode.heatmap
    heatmap_field: HeatmapField = HeatmapField.divergence
    transparent_objects: list = None
    invisible_objects: list = None
    color_alpha: float = 1.0
    deformable_mesh_view: bool = False


def _title(s: str) -> str:
    return s.replace("_", " ").title()

@ti.func
def _clamp(x, a, b):
    return ti.min(ti.max(x, a), b)

@ti.func
def _lerp(a: ti.types.vector(3, ti.f32), b: ti.types.vector(3, ti.f32), t: ti.f32):
    return a * (1.0 - t) + b * t

@ti.func
def _heat_vel_color(s: ti.f32) -> ti.types.vector(3, ti.f32):
    # blue (0,0,1) -> white (1,1,1)
    c0 = ti.Vector([0.0, 0.0, 1.0])
    c1 = ti.Vector([1.0, 1.0, 1.0])
    return _lerp(c0, c1, s)

@ti.func
def _heat_div_color(s: ti.f32) -> ti.types.vector(3, ti.f32):
    # blue -> white -> red
    cB = ti.Vector([0.0, 0.0, 1.0])
    cW = ti.Vector([1.0, 1.0, 1.0])
    cR = ti.Vector([1.0, 0.0, 0.0])
    ret = cW
    if s < 0.5:
        ret = _lerp(cB, cW, s * 2.0)
    else:
        ret = _lerp(cW, cR, (s - 0.5) * 2.0)
    return ret

@ti.func
def _heat_den_color(s: ti.f32) -> ti.types.vector(3, ti.f32):
    cW = ti.Vector([1.0, 1.0, 1.0])
    cR = ti.Vector([1.0, 0.0, 0.0])
    ret = cW
    if s >= 0.5:
        ret = _lerp(cW, cR, (s - 0.5) * 2.0)
    return ret

@ti.data_oriented
class VisualizationEngine:
    def __init__(self, ps, config, settings: VisualizationSettings):
        self.ps = ps
        self.settings = settings

        x_max, y_max, z_max = config.get_cfg("domainEnd")
        self.box_anchors = ti.Vector.field(3, dtype=ti.f32, shape=8)
        self.box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
        self.box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
        self.box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
        self.box_anchors[3] = ti.Vector([x_max, y_max, 0.0])
        self.box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
        self.box_anchors[5] = ti.Vector([0.0, y_max, z_max])
        self.box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
        self.box_anchors[7] = ti.Vector([x_max, y_max, z_max])
        self.box_lines_indices = ti.field(int, shape=(2 * 12))
        for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
            self.box_lines_indices[i] = val

        # self.last_rgba = None               # (N,4) float
        # self.last_scalar_raw = None         # (N,) float
        # self.last_scalar_norm = None        # (N,) float
        # self.last_field = self.settings.heatmap_field

        # mesh caches
        self._solid_face_indices = {}     # obj_id -> ti.field(int, 3*face_count)
        self._rigid_vert_fields = {}      # obj_id -> ti.Vector.field(3, ti.f32, |V|)
        self._rigid_face_fields = {}      # obj_id -> ti.field(int, 3*|F|)
        self._rigid_rest_vertices = {}    # obj_id -> np.ndarray(|V|,3)

        self.scalar_raw  = ti.field(dtype=ti.f32, shape=self.ps.particle_max_num)
        self.scalar_norm = ti.field(dtype=ti.f32, shape=self.ps.particle_max_num)
        self.is_transparent = ti.field(dtype=ti.i32, shape=self.ps.particle_max_num)  # particle-wise mask

        # Normalize parameters
        self.vmin_v, self.vmax_v = 0.0, 1.5
        self.vmin_div, self.vmax_div = 0.0, 5.0
        self.vmin_den, self.vmax_den = -30.0, 30.0

        self.max_object_id = 10
        self.transparent_obj_table = ti.field(dtype=ti.i32, shape=self.max_object_id)  # 0/1

        # self.norm_v = Normalize(vmin=0.0, vmax=1.5, clip=True)
        # self.norm_div = Normalize(vmin=0.0, vmax=5.0, clip=True)
        # self.norm_density = Normalize(vmin=-30.0, vmax=30.0, clip=True)

        # self.cmap_vel = LinearSegmentedColormap.from_list("heat_vel", ["blue", "white"])
        # self.cmap_div = LinearSegmentedColormap.from_list("heat_div", ["blue", "white", "red"])
        # self.cmap_den = LinearSegmentedColormap.from_list("heat_den", ["white", "white", "red"])
        
        self.pin_vertices = None
        self.pin_indices = None

        self.prepare_mesh_fields()


    def set_pin_lines(self, vertices_np, indices_np):
        if vertices_np is None or indices_np is None or len(vertices_np) == 0:
            self.pin_vertices, self.pin_indices = None, None
            return
        nV = int(vertices_np.shape[0]); nI = int(indices_np.shape[0])
        self.pin_vertices = ti.Vector.field(3, dtype=ti.f32, shape=nV)
        self.pin_indices = ti.field(int, shape=nI)
        self.pin_vertices.from_numpy(vertices_np.astype(np.float32))
        self.pin_indices.from_numpy(indices_np.astype(np.int32))


    # def copy_to_vis_buffer(self, invisible_objects=[]):
    #     # Always clear buffers to avoid rendering inactive/emitted stale particles
    #     self.ps.x_vis_buffer.fill(1000.0)  # move non-active far away
    #     self.ps.color_vis_buffer.fill(0.0)
    #     for obj_id in self.ps.object_collection:
    #         if obj_id not in invisible_objects:
    #             self._copy_to_vis_buffer(obj_id)


    # @ti.kernel
    # def _copy_to_vis_buffer(self, obj_id: int):
    #     assert self.ps.GGUI
    #     # Only copy active particles
    #     for i in range(self.ps.particle_num[None]):
    #         if self.ps.object_id[i] == obj_id:
    #             self.ps.x_vis_buffer[i] = self.ps.x[i]
    #             self.ps.color_vis_buffer[i] = self.ps.color[i] / 255.0


    def update_buffers(self):
        self._update_transparent_table()
        self._build_transparent_mask_and_copy_vis()

        if self.settings.heatmap_field == HeatmapField.velocity:
            field_mode = 0
        elif self.settings.heatmap_field == HeatmapField.divergence:
            field_mode = 1
        else:
            field_mode = 2

        self._compute_scalar_and_normalize(field_mode)

        # Apply colors (heatmap/original, transparency)
        use_heatmap = 1 if self.settings.color_mode == ColorMode.heatmap else 0
        alpha = float(self.settings.color_alpha or 1.0)
        self._apply_colors(field_mode, use_heatmap, alpha)

        # Hide solids/rigids in mesh view
        self._apply_deformable_mesh_view_hiding(1 if self.settings.deformable_mesh_view else 0)

        # Cache
        N_active = int(self.ps.particle_num[None])
        self._N_active = N_active
        self._object_id_np = self.ps.object_id.to_numpy()[:N_active]
        self._material_np = self.ps.material.to_numpy()[:N_active]
        self._default_colors_np = self.ps.color_vis_buffer.to_numpy()[:N_active]  # 0~1

        # self.last_rgba = self.ps.color_heat_map.to_numpy()[:N_active]
        self._render_colors = self.ps.color_heat_map



    def draw(self, scene, canvas, background_color=(0, 0, 0)):
        bg = tuple(c / 255.0 for c in background_color) if max(background_color) > 1 else background_color
        canvas.set_background_color(bg)
        scene.particles(self.ps.x_vis_buffer, radius=self.ps.particle_radius, per_vertex_color=self._render_colors)

        if self.settings.deformable_mesh_view:
            # scene.mesh(self.ps.x_s, indices=self.ps.surface_faces, color=(0.99, 0.68, 0.28), two_sided=True)
            for obj_id, obj in self.ps.object_collection.items():
                f_field = self._solid_face_indices.get(obj_id, None)
                if f_field is not None:
                    col = np.array(obj.get("color", [200,80,80]))/255.0
                    scene.mesh(self.ps.x_s, indices=f_field, color=tuple(col.tolist()), two_sided=True)
            for r_body_id, v_field in self._rigid_vert_fields.items():
                f_field = self._rigid_face_fields.get(r_body_id, None)
                if v_field is not None and f_field is not None:
                    rb = self.ps.object_collection.get(int(r_body_id), None)
                    col = np.array(rb.get("color", [160,160,160]))/255.0
                    scene.mesh(v_field, indices=f_field, color=tuple(col.tolist()), two_sided=True)

                    # print(self.ps.surface_faces.to_numpy())

        scene.lines(self.box_anchors, indices=self.box_lines_indices, color=(0.99, 0.68, 0.28, 1.0), width=1.0)
        if self.pin_vertices is not None and self.pin_indices is not None:
            scene.lines(self.pin_vertices, indices=self.pin_indices, color=(1.0, 0.2, 0.2, 1.0), width=1.5)
        canvas.scene(scene)


    def render_ui(self, w, gui) -> None:
        modes = list(ColorMode)
        cur_mode_idx = modes.index(self.settings.color_mode) + 1
        cur_mode_idx = w.slider_int("visualization mode", cur_mode_idx, 1, len(modes))
        self.settings.color_mode = modes[cur_mode_idx - 1]

        gui.text("")
        self.settings.deformable_mesh_view = w.checkbox("mesh view", self.settings.deformable_mesh_view)
        self.settings.deformable_mesh_view = bool(self.settings.deformable_mesh_view)

        if self.settings.color_mode == ColorMode.heatmap:
            gui.text("Heatmap Mode")
            fields = list(HeatmapField)
            cur_hm_idx = fields.index(self.settings.heatmap_field) + 1
            cur_hm_idx = w.slider_int("heatmap type", cur_hm_idx, 1, len(fields))
            self.settings.heatmap_field = fields[cur_hm_idx - 1]
            gui.text(_title(self.settings.heatmap_field.value))
        else:
            gui.text("Original Colors Mode")

        gui.text("")  # Spacer
        gui.text("Transparency Controls:")
        if self.settings.transparent_objects:
            invisible = (self.settings.color_alpha < 0.01)
            invisible = w.checkbox("invisible", invisible)
            self.settings.color_alpha = 0.0 if invisible else 0.2
            gui.text(f"Transparent objects: {self.settings.transparent_objects}")
            gui.text(f"Current alpha: {self.settings.color_alpha:.2f}")
        else:
            gui.text("No transparent objects configured")


    def prepare_mesh_fields(self):
        for obj_id, obj in self.ps.object_collection.items():
            if obj.get("surfaceFaceCount", 0) > 0:
                self._ensure_solid_face_field(int(obj_id), obj)
        for r_body_id in self.ps.object_id_rigid_body:
            self._ensure_rigid_mesh_fields(int(r_body_id))


    def _rigid_transform_from_particles(self, obj_id: int):
        rb = self.ps.object_collection.get(int(obj_id), None)
        is_dynamic = bool(rb and rb.get("isDynamic", False))
        if not is_dynamic:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        N = int(self.ps.particle_num[None])
        if N <= 0:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        obj_ids = self.ps.object_id.to_numpy()[:N]
        mats = self.ps.material.to_numpy()[:N]
        mask = (obj_ids == int(obj_id)) & (mats == self.ps.material_rigid)
        if not np.any(mask):
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        X0 = self.ps.x0.to_numpy()[:N][mask].astype(np.float32)
        X  = self.ps.x.to_numpy()[:N][mask].astype(np.float32)
        c0 = X0.mean(axis=0)
        c  = X.mean(axis=0)
        P, Q = X0 - c0, X - c

        R = np.eye(3, dtype=np.float32)
        if P.shape[0] >= 3:
            H = P.T @ Q
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
        t = c - R @ c0
        return R.astype(np.float32), t.astype(np.float32)

    def _ensure_solid_face_field(self, obj_id: int, obj: dict):
        fc = int(obj.get("surfaceFaceCount", 0))
        if fc <= 0:
            return None
        if obj_id in self._solid_face_indices:
            return self._solid_face_indices[obj_id]
        fs = int(obj.get("surfaceFaceOffset", 0))
        F_all = self.ps.surface_faces.to_numpy().astype(np.int32)
        F_sub = F_all[3 * fs : 3 * (fs + fc)]
        f_field = ti.field(int, shape=int(F_sub.shape[0]))
        f_field.from_numpy(F_sub)
        self._solid_face_indices[obj_id] = f_field
        return f_field

    def _ensure_rigid_mesh_fields(self, obj_id: int):
        rb = self.ps.object_collection.get(int(obj_id), None)
        if rb is None:
            return None, None
        mesh_rest = rb.get("mesh", None)
        if mesh_rest is None:
            return None, None

        V_rest = np.asarray(mesh_rest.vertices, dtype=np.float32)
        F = np.asarray(mesh_rest.faces, dtype=np.int32).reshape(-1)

        if obj_id not in self._rigid_face_fields:
            f_field = ti.field(int, shape=int(F.shape[0]))
            f_field.from_numpy(F.astype(np.int32))
            self._rigid_face_fields[obj_id] = f_field

        if obj_id not in self._rigid_vert_fields:
            v_field = ti.Vector.field(3, dtype=ti.f32, shape=int(V_rest.shape[0]))
            self._rigid_vert_fields[obj_id] = v_field
            self._rigid_rest_vertices[obj_id] = V_rest

        is_dynamic = bool(rb.get("isDynamic", False))
        if is_dynamic:
            Rm, tm_vec = self._rigid_transform_from_particles(int(obj_id))
            V_tr = self._rigid_rest_vertices[obj_id] @ Rm.T + tm_vec[None, :]
        else:
            V_tr = self._rigid_rest_vertices[obj_id]

        self._rigid_vert_fields[obj_id].from_numpy(V_tr.astype(np.float32))
        return self._rigid_vert_fields[obj_id], self._rigid_face_fields[obj_id]


    def _update_transparent_table(self):
        # 테이블 초기화
        self.transparent_obj_table.fill(0)
        if not self.settings.transparent_objects:
            return
        # 안전 범위 내에서만 세팅
        for oid in self.settings.transparent_objects:
            oid_i = int(oid)
            if 0 <= oid_i < self.max_object_id:
                self.transparent_obj_table[oid_i] = 1


    @ti.kernel
    def _build_transparent_mask_and_copy_vis(self):
        n = self.ps.particle_num[None]

        for i in range(n):
            self.ps.x_vis_buffer[i] = self.ps.x[i]
            self.ps.color_vis_buffer[i] = self.ps.color[i] / 255.0
            self.is_transparent[i] = 0
            oid = self.ps.object_id[i]
            if 0 <= oid and oid < self.max_object_id and self.transparent_obj_table[oid] == 1:
                self.is_transparent[i] = 1

        for i in range(n, self.ps.particle_max_num):
            self.ps.x_vis_buffer[i] = ti.Vector([1000.0, 1000.0, 1000.0])
            self.ps.color_vis_buffer[i] = ti.Vector([0.0, 0.0, 0.0, 0.0])


    @ti.kernel
    def _compute_scalar_and_normalize(self, field_mode: ti.i32):
        # field_mode: 0=velocity, 1=divergence, 2=density
        n = self.ps.particle_num[None]
        for i in range(n):
            val = 0.0
            if field_mode == 0:
                v = self.ps.v[i]
                val = ti.sqrt(v.dot(v))
                val = (val - self.vmin_v) / (self.vmax_v - self.vmin_v + 1e-8)
            elif field_mode == 1:
                val = self.ps.divergence[i]
                val = (val - self.vmin_div) / (self.vmax_div - self.vmin_div + 1e-8)
            else:  # density diff
                val = (self.ps.density[i] - self.ps.density0[i])
                val = (val - self.vmin_den) / (self.vmax_den - self.vmin_den + 1e-8)

            val = _clamp(val, 0.0, 1.0)
            self.scalar_raw[i] = val
            self.scalar_norm[i] = val


    @ti.kernel
    def _apply_deformable_mesh_view_hiding(self, enable: ti.i32):
        if enable == 1:
            n = self.ps.particle_num[None]
            for i in range(n):
                mat = self.ps.material[i]
                if (mat == self.ps.material_solid) or (mat == self.ps.material_rigid):
                    self.ps.x_vis_buffer[i] = ti.Vector([1000.0, 1000.0, 1000.0])


    @ti.kernel
    def _apply_colors(self, field_mode: ti.i32, use_heatmap: ti.i32, alpha: ti.f32):
        # use_heatmap: 1=heatmap, 0=original
        # alpha: settings.color_alpha
        n = self.ps.particle_num[None]
        for i in range(n):
            col = self.ps.color_vis_buffer[i]

            apply_heatmap = use_heatmap == 1 and self.ps.is_static_rigid(i) == 0
            if apply_heatmap:
                s = self.scalar_norm[i]
                rgb = ti.Vector([1.0, 1.0, 1.0])
                if field_mode == 0:
                    rgb = _heat_vel_color(s)
                elif field_mode == 1:
                    rgb = _heat_div_color(s)
                else:
                    rgb = _heat_den_color(s)
                col = ti.Vector([rgb[0], rgb[1], rgb[2], 1.0])

            # Adjust transparency of transparent objects
            if self.is_transparent[i] == 1:
                if alpha < 0.01:
                    # Make it effectively invisible: move far away and set alpha=0
                    self.ps.x_vis_buffer[i] = ti.Vector([1000.0, 1000.0, 1000.0])
                    col[3] = 0.0
                else:
                    # heatmap mode: α=alpha, original mode: α=alpha*0.5
                    shadow_alpha = alpha if use_heatmap == 1 else alpha * 0.5
                    col[3] = shadow_alpha

            self.ps.color_heat_map[i] = col



    def get_heatmap_attributes(self):
        if not hasattr(self, "_render_colors") or self._render_colors is None:
            self.update_buffers()

        N = int(self.ps.particle_num[None])

        rgba_np          = self.ps.color_heat_map.to_numpy()[:N].copy()
        object_id_np     = self.ps.object_id.to_numpy()[:N].copy()
        material_np      = self.ps.material.to_numpy()[:N].copy()
        default_colors_np= self.ps.color_vis_buffer.to_numpy()[:N].copy()  # 0~1

        return {
            "N": N,
            "object_id": object_id_np,
            "material": material_np,
            "default_colors": default_colors_np,
            "rgba": rgba_np,
        }