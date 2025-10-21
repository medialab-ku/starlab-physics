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

        self.last_rgba = None               # (N,4) float
        self.last_scalar_raw = None         # (N,) float
        self.last_scalar_norm = None        # (N,) float
        self.last_field = self.settings.heatmap_field

        self.norm_v = Normalize(vmin=0.0, vmax=1.5, clip=True)
        self.norm_div = Normalize(vmin=0.0, vmax=5.0, clip=True)
        self.norm_density = Normalize(vmin=-50.0, vmax=50.0, clip=True)

        self.cmap_vel = LinearSegmentedColormap.from_list("heat_vel", ["blue", "white"])
        self.cmap_div = LinearSegmentedColormap.from_list("heat_div", ["blue", "white", "red"])
        self.cmap_den = LinearSegmentedColormap.from_list("heat_den", ["blue", "white", "red"])


    def copy_to_vis_buffer(self, invisible_objects=[]):
        # Always clear buffers to avoid rendering inactive/emitted stale particles
        self.ps.x_vis_buffer.fill(1000.0)  # move non-active far away
        self.ps.color_vis_buffer.fill(0.0)
        for obj_id in self.ps.object_collection:
            if obj_id not in invisible_objects:
                self._copy_to_vis_buffer(obj_id)


    @ti.kernel
    def _copy_to_vis_buffer(self, obj_id: int):
        assert self.ps.GGUI
        # Only copy active particles
        for i in range(self.ps.particle_num[None]):
            if self.ps.object_id[i] == obj_id:
                self.ps.x_vis_buffer[i] = self.ps.x[i]
                self.ps.color_vis_buffer[i] = self.ps.color[i] / 255.0

    def update_buffers(self):
        self.copy_to_vis_buffer(invisible_objects=(self.settings.invisible_objects or []))
        
        N_active = int(self.ps.particle_num[None])
           
        v_np = self.ps.v.to_numpy()
        density_np = self.ps.density.to_numpy()
        density0_np = self.ps.density0.to_numpy()
        div_np = self.ps.divergence.to_numpy()
        material_np = self.ps.material.to_numpy()
        dynamic_mask = self.ps.is_dynamic.to_numpy()

        v_norm = np.linalg.norm(v_np, axis=1)
        density_diff = density_np - density0_np

        if self.settings.heatmap_field == HeatmapField.velocity:
            scalar_norm = self.norm_v(v_norm)
            rgba = self.cmap_vel(scalar_norm)
        elif self.settings.heatmap_field == HeatmapField.divergence:
            scalar_norm = self.norm_div(div_np)
            rgba = self.cmap_div(scalar_norm)
        else:  # density
            scalar_norm = self.norm_density(density_diff)
            rgba = self.cmap_den(scalar_norm)

        default_colors = self.ps.color_vis_buffer.to_numpy()

        # TODO: fix redundant logic for heatmap & original color mode
        if self.settings.color_mode == ColorMode.heatmap: 
            heat_map_colors = default_colors.copy()
            fluid_mask = (material_np == self.ps.material_fluid)
            show_mask = np.logical_or(fluid_mask, dynamic_mask)
            heat_map_colors[show_mask] = rgba[show_mask]

            # TODO: unify invisible objects & transparent objects
            object_id_np = self.ps.object_id.to_numpy()
            for obj_id in (self.settings.transparent_objects or []):
                obj_mask = (object_id_np == obj_id)
                heat_map_colors[obj_mask, 3] = self.settings.color_alpha

            if self.settings.color_alpha < 0.01:
                transparent_mask = np.zeros_like(heat_map_colors[:, 0], dtype=bool)
                for obj_id in (self.settings.transparent_objects or []):
                    transparent_mask |= (object_id_np == obj_id)
                transparent_positions = self.ps.x.to_numpy()
                transparent_positions[transparent_mask] = [1000.0, 1000.0, 1000.0]
                self.ps.x_vis_buffer.from_numpy(transparent_positions)
                heat_map_colors[transparent_mask, 3] = 0.0
            else:
                shadow_alpha = self.settings.color_alpha
                for obj_id in (self.settings.transparent_objects or []):
                    obj_mask = (object_id_np == obj_id)
                    heat_map_colors[obj_mask, 3] = shadow_alpha

            self.ps.color_heat_map.from_numpy(heat_map_colors)
            render_colors = self.ps.color_heat_map

        else:
            original_colors = default_colors.copy()
            object_id_np = self.ps.object_id.to_numpy()
            for obj_id in (self.settings.transparent_objects or []):
                obj_mask = (object_id_np == obj_id)
                original_colors[obj_mask, 3] = self.settings.color_alpha

            if self.settings.color_alpha < 0.01:
                transparent_mask = np.zeros_like(original_colors[:, 0], dtype=bool)
                for obj_id in (self.settings.transparent_objects or []):
                    transparent_mask |= (object_id_np == obj_id)
                transparent_positions = self.ps.x.to_numpy()
                transparent_positions[transparent_mask] = [1000.0, 1000.0, 1000.0]
                self.ps.x_vis_buffer.from_numpy(transparent_positions)
                original_colors[transparent_mask, 3] = 0.0
            else:
                shadow_alpha = self.settings.color_alpha * 0.5
                for obj_id in (self.settings.transparent_objects or []):
                    obj_mask = (object_id_np == obj_id)
                    original_colors[obj_mask, 3] = shadow_alpha

            self.ps.color_heat_map.from_numpy(original_colors)
            render_colors = self.ps.color_heat_map

        if self.settings.deformable_mesh_view:
            pos = self.ps.x_vis_buffer.to_numpy()
            solid_mask = (material_np == self.ps.material_solid)
            pos[solid_mask] = [1000.0, 1000.0, 1000.0]
            self.ps.x_vis_buffer.from_numpy(pos)


        # Cache for export
        self.last_rgba = rgba
        self._N_active = N_active
        self._object_id_np = object_id_np[:N_active]
        self._material_np = material_np[:N_active]
        self._default_colors_np = default_colors[:N_active]  # 0~1
        # Render handle
        self._render_colors = render_colors


    def draw(self, scene, canvas, background_color=(0, 0, 0)):
        bg = tuple(c / 255.0 for c in background_color) if max(background_color) > 1 else background_color
        canvas.set_background_color(bg)
        scene.particles(self.ps.x_vis_buffer, radius=self.ps.particle_radius, per_vertex_color=self._render_colors)

        if self.settings.deformable_mesh_view:
            scene.mesh(self.ps.x_s, indices=self.ps.surface_faces, color=(0.99, 0.68, 0.28), two_sided=True)
            # print(self.ps.surface_faces.to_numpy())

        scene.lines(self.box_anchors, indices=self.box_lines_indices, color=(0.99, 0.68, 0.28, 1.0), width=1.0)
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

    
    # visualization.py - VisualizationEngine 클래스 내부
    def get_heatmap_attributes(self):
        if self.last_rgba is None:
            self.update_buffers()
        return {
            "N": self._N_active,
            "object_id": self._object_id_np,
            "material": self._material_np,
            "default_colors": self._default_colors_np,
            "rgba": self.last_rgba,
        }