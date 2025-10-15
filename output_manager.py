import os, time
import numpy as np
import taichi as ti
import trimesh as tm
from dataclasses import dataclass
from enum import Enum

# OutputManager: get_config
# OutputConfig
# ExportFormat

class ExportFormat(str, Enum):
    ply = "ply"
    vtk = "vtk"        # TODO
    partio = "partio"  # TODO

@dataclass
class OutputConfig:
    export_particles: bool = True
    selected_format: ExportFormat = ExportFormat.ply
    export_fluid_particles: bool = True
    export_rigid_particles: bool = False
    export_mesh_obj: bool = False
    frame_interval: int = 20
    include_heatmap_attributes: bool = True
    end_frame: int = 8000


class OutputManager:
    def __init__(self, scene_name: str, cfg: OutputConfig,
                 root_base: str = "output", timestamp: str | None = None):
        self.cfg = cfg
        self.scene_name = scene_name
        self.timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(root_base, scene_name, self.timestamp)
        self.ply_dir = self.run_dir
        self.obj_dir = os.path.join(self.run_dir, "mesh_obj")
        self._cnt_ply = 0
        self._cnt_obj = 0
        self._warned_formats = set()
        self._ensure_dirs()

    def _ensure_dirs(self):
        if ExportFormat.ply in self.cfg.selected_format:
            os.makedirs(self.ply_dir, exist_ok=True)
        if self.cfg.export_mesh_obj:
            os.makedirs(self.obj_dir, exist_ok=True)


    def get_config(self) -> OutputConfig:
        return self.cfg

    def set_config(self, cfg: OutputConfig) -> None:
        self.cfg = cfg
        self._ensure_dirs()


    def on_step(self, frame_idx: int, ps, viz_engine):
        if frame_idx % self.cfg.frame_interval != 0:
            return

        if self.cfg.export_particles:
            export_data = viz_engine.get_heatmap_attributes() if self.cfg.include_heatmap_attributes else None
            export_format = self.cfg.selected_format
            if export_format == ExportFormat.ply:
                self._export_particles_ply(ps, export_data)
            else:
                if export_format not in self._warned_formats:
                    print(f"[Export] {export_format.value} Not implemented yet")
                    self._warned_formats.add(export_format)
                    # TODO: VTK and Partio are not implemented yet
                    # elif ExportFormat.vtk in self.cfg.export_formats:
                    #     self._export_particles_vtk(ps, heat)
                    # elif ExportFormat.partio in self.cfg.export_formats:
                    #     self._export_particles_partio(ps, heat)

        if self.cfg.export_mesh_obj and len(getattr(ps, "object_id_rigid_body", [])) > 0:
            self._export_mesh_obj(ps)


    def _export_particles_ply(self, ps, export_data):
        N_active = int(ps.particle_num[None])
        if N_active <= 0:
            return

        if export_data is None:
            object_id_np = ps.object_id.to_numpy()[:N_active]
            material_np = ps.material.to_numpy()[:N_active]
            default_colors = ps.color_vis_buffer.to_numpy()[:N_active]
            rgba = None
        else:
            object_id_np = export_data["object_id"]
            material_np = export_data["material"]
            default_colors = export_data["default_colors"]
            rgba = export_data.get("rgba", None)

        pos_all = ps.x.to_numpy()[:N_active]

        for obj_id in ps.object_collection:
            mask = (object_id_np == int(obj_id))
            if not np.any(mask):
                continue

            is_fluid = bool(np.any(material_np[mask] == ps.material_fluid))
            if is_fluid and not self.cfg.export_fluid_particles:
                continue
            if (not is_fluid) and not self.cfg.export_rigid_particles:
                continue

            pos = pos_all[mask]
            if len(pos) == 0:
                continue

            if rgba is not None:
                colors = rgba[mask][:, :3]
            else:
                colors = default_colors[mask][:, :3]

            prefix = os.path.join(self.ply_dir, f"particle_object_{int(obj_id)}.ply")
            writer = ti.tools.PLYWriter(num_vertices=len(pos))
            writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
            writer.add_vertex_color(colors[:, 0], colors[:, 1], colors[:, 2])
            writer.export_frame_ascii(self._cnt_ply, prefix)

        self._cnt_ply += 1


    def _export_mesh_obj(self, ps):
        for r_body_id in ps.object_id_rigid_body:
            rb = ps.object_collection.get(r_body_id, None)
            if rb is None:
                continue
            mesh_rest = rb.get("mesh", None)
            if mesh_rest is None:
                continue
            Rm, tm_vec = self._rigid_transform_from_particles(ps, int(r_body_id))
            V_rest = np.asarray(mesh_rest.vertices, dtype=np.float32)
            V_tr = V_rest @ Rm.T + tm_vec[None, :]
            F = np.asarray(mesh_rest.faces) if hasattr(mesh_rest, "faces") else None
            mesh_out = tm.Trimesh(vertices=V_tr, faces=F, process=False)
            out_path = os.path.join(self.obj_dir, f"obj_{int(r_body_id)}_{self._cnt_obj:06}.obj")
            mesh_out.export(out_path)

        self._cnt_obj += 1        


    def _rigid_transform_from_particles(self, ps, obj_id: int):
        N = int(ps.particle_num[None])
        if N <= 0:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        obj_ids = ps.object_id.to_numpy()[:N]
        mats = ps.material.to_numpy()[:N]
        mask = (obj_ids == obj_id) & (mats == ps.material_solid)
        if not np.any(mask):
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        X0 = ps.x_0.to_numpy()[:N][mask].astype(np.float32)
        X = ps.x.to_numpy()[:N][mask].astype(np.float32)
        c0 = X0.mean(axis=0)
        c = X.mean(axis=0)
        P = X0 - c0
        Q = X - c
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

    
    def render_ui(self, w, gui, ps):
        cfg = self.cfg

        # Export particles
        cfg.export_particles = w.checkbox("Export particles", bool(cfg.export_particles))
        if cfg.export_particles:
            # Format slider (PLY / VTK / Partio)
            formats = [ExportFormat.ply, ExportFormat.vtk, ExportFormat.partio]
            fmt_idx = formats.index(cfg.selected_format) + 1
            fmt_idx = w.slider_int("format (1:ply,2:vtk,3:partio)", fmt_idx, 1, len(formats))
            cfg.selected_format = formats[fmt_idx - 1]

            # TODO: VTK and Partio are not implemented yet
            if cfg.selected_format in (ExportFormat.vtk, ExportFormat.partio):
                gui.text("Not implemented yet")

            cfg.export_fluid_particles = w.checkbox("Fluid", bool(cfg.export_fluid_particles))
            cfg.export_rigid_particles = w.checkbox("Rigid", bool(cfg.export_rigid_particles))
            if not cfg.export_fluid_particles and not cfg.export_rigid_particles:
                cfg.export_fluid_particles = True

            cfg.frame_interval = w.slider_int("frame interval", int(cfg.frame_interval), 1, 200)
            cfg.include_heatmap_attributes = w.checkbox("include heatmap attrs", bool(cfg.include_heatmap_attributes))
            cfg.end_frame = w.slider_int("end frame", int(cfg.end_frame), 0, int(3e4))

        # Export mesh (OBJ)
        has_rigid = len(getattr(ps, "object_id_rigid_body", [])) > 0
        if has_rigid:
            cfg.export_mesh_obj = w.checkbox("Export mesh (OBJ)", bool(cfg.export_mesh_obj))
        else:
            cfg.export_mesh_obj = False

        self.cfg = cfg
        self._ensure_dirs()