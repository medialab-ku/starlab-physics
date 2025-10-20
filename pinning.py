import taichi as ti
import numpy as np
from typing import Dict, List, Tuple, Optional


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def _rodrigues(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    # axis-angle -> 3x3 (row-vector convention)
    a = _normalize(axis)
    x, y, z = a
    c = float(np.cos(angle_rad)); s = float(np.sin(angle_rad)); C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C   ],
    ], dtype=np.float32)


@ti.data_oriented
class ParticlePinning:
    def __init__(self, particle_system) -> None:
        self.ps = particle_system
        self.dim = int(self.ps.dim)
        assert self.dim == 3

    def apply(self, scene_data: Dict) -> Dict[str, np.ndarray]:
        """
        Build pin segments from Pin.x_ranges/y_ranges (OR), freeze those particles,
        and return guide line geometry. Also store per-object Pin.kinematic config.
        """
        N = int(self.ps.particle_num[None])
        object_id_np = self.ps.object_id.to_numpy()[:N]
        x0_np = self.ps.x0.to_numpy()[:N]
        material_np = self.ps.material.to_numpy()[:N]

        vertices: List[np.ndarray] = []
        indices: List[Tuple[int, int]] = []
        pinned_global_indices: List[int] = []

        solids = scene_data.get("solid_bodies", [])
        obj_map = scene_data.get("object_map", {})

        for solid in solids:
            obj_id = int(solid["objectId"])
            if obj_id not in obj_map:
                continue
            cfg = obj_map[obj_id]
            pin_cfg = dict(cfg.get("Pin", {}) or {})
            if not pin_cfg:
                continue

            # rotation (local->world), COM from mesh rest
            axis = np.array(cfg.get("rotationAxis", [0, 0, 1]), dtype=np.float32)
            ang_deg = float(cfg.get("rotationAngle", 0.0))
            ang_rad = ang_deg / 360.0 * 2.0 * np.pi
            R = _rodrigues(axis, ang_rad)
            COM = np.array(cfg.get("restCenterOfMass", [0, 0, 0]), dtype=np.float32)

            # object particles (solid only)
            mask_obj = (object_id_np == obj_id) & (material_np == self.ps.material_solid)
            if not np.any(mask_obj):
                # still store kinematic config if any
                self.obj_kin[obj_id] = pin_cfg.get("kinematic")
                continue

            idxs_global = np.nonzero(mask_obj)[0].astype(np.int32)
            P_w = x0_np[mask_obj]  # world (rest)

            # to local (row-vector form): local = (world - COM) @ R
            P_local = (P_w - COM) @ R
            local_min = P_local.min(axis=0)
            local_max = P_local.max(axis=0)
            local_size = np.maximum(local_max - local_min, 1e-12)
            P_local_shift = P_local - local_min  # [0, size]

            normalized_pin = bool(pin_cfg.get("normalized", True))

            def to_abs_range_list(ranges: Optional[List[List[float]]], axis_i: int) -> List[Tuple[float, float]]:
                if not ranges:
                    return []
                rs = []
                for lo, hi in ranges:
                    lo_abs = float(lo) * float(local_size[axis_i]) if normalized_pin else float(lo)
                    hi_abs = float(hi) * float(local_size[axis_i]) if normalized_pin else float(hi)
                    lo_a, hi_a = (min(lo_abs, hi_abs), max(lo_abs, hi_abs))
                    lo_a = max(0.0, lo_a)
                    hi_a = min(float(local_size[axis_i]), hi_a)
                    rs.append((lo_a, hi_a))
                return rs

            x_ranges = to_abs_range_list(pin_cfg.get("x_ranges"), 0)
            y_ranges = to_abs_range_list(pin_cfg.get("y_ranges"), 1)

            eps = 1e-9
            pin_mask_local = np.zeros(P_local_shift.shape[0], dtype=bool)
            seg_records_local: List[Dict] = []

            # x segments
            if len(x_ranges) > 0:
                xvals = P_local_shift[:, 0]
                for (lo_a, hi_a) in x_ranges:
                    seg_m = (xvals >= lo_a - eps) & (xvals <= hi_a + eps)
                    if np.any(seg_m):
                        pin_mask_local |= seg_m
                        seg_records_local.append({
                            "axis": "x",
                            "range_abs": (lo_a, hi_a),
                            "indices_local": np.nonzero(seg_m)[0].astype(np.int32),
                        })
                    # guide lines: planes x=lo, x=hi
                    vertices, indices = self._append_plane_lines(
                        vertices, indices, axis_idx=0, t_abs=lo_a,
                        local_min=local_min, local_max=local_max, R=R, COM=COM)
                    vertices, indices = self._append_plane_lines(
                        vertices, indices, axis_idx=0, t_abs=hi_a,
                        local_min=local_min, local_max=local_max, R=R, COM=COM)

            # y segments
            if len(y_ranges) > 0:
                yvals = P_local_shift[:, 1]
                for (lo_a, hi_a) in y_ranges:
                    seg_m = (yvals >= lo_a - eps) & (yvals <= hi_a + eps)
                    if np.any(seg_m):
                        pin_mask_local |= seg_m
                        seg_records_local.append({
                            "axis": "y",
                            "range_abs": (lo_a, hi_a),
                            "indices_local": np.nonzero(seg_m)[0].astype(np.int32),
                        })
                    # guide lines: planes y=lo, y=hi
                    vertices, indices = self._append_plane_lines(
                        vertices, indices, axis_idx=1, t_abs=lo_a,
                        local_min=local_min, local_max=local_max, R=R, COM=COM)
                    vertices, indices = self._append_plane_lines(
                        vertices, indices, axis_idx=1, t_abs=hi_a,
                        local_min=local_min, local_max=local_max, R=R, COM=COM)

            if np.any(pin_mask_local):
                pinned_indices_obj = idxs_global[pin_mask_local]
                pinned_global_indices.extend(pinned_indices_obj.tolist())


        # freeze pinned
        pinned_np = np.array(pinned_global_indices, dtype=np.int32)
        if pinned_np.size > 0:
            self._set_static(pinned_np.shape[0], pinned_np)

        if len(vertices) == 0:
            return {"vertices": None, "indices": None}

        V = np.vstack(vertices).astype(np.float32)
        E = np.array(indices, dtype=np.int32).reshape(-1, 2).flatten()
        return {"vertices": V, "indices": E}

    def _append_plane_lines(
        self,
        vertices: List[np.ndarray],
        indices: List[Tuple[int, int]],
        axis_idx: int,
        t_abs: float,
        local_min: np.ndarray,
        local_max: np.ndarray,
        R: np.ndarray,
        COM: np.ndarray
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        lm, lx = local_min, local_max
        ls = lx - lm
        t_abs = float(np.clip(t_abs, 0.0, float(ls[axis_idx])))

        # enlarge guide rectangle orthogonal to the pinned axis
        pr = float(self.ps.particle_radius)
        pad_big = 6.0 * pr     # more visible
        pad_small = 3.0 * pr

        if axis_idx == 0:
            # x-plane -> enlarge along y,z
            y0, y1 = -pad_big, float(ls[1]) + pad_big
            z0, z1 = -pad_big, float(ls[2]) + pad_big
            rect_local = np.array([
                [t_abs, y0, z0],
                [t_abs, y0, z1],
                [t_abs, y1, z1],
                [t_abs, y1, z0],
            ], dtype=np.float32)
        else:
            # y-plane -> enlarge along x,z (slightly smaller if desired)
            x0, x1 = -pad_small, float(ls[0]) + pad_small
            z0, z1 = -pad_small, float(ls[2]) + pad_small
            rect_local = np.array([
                [x0, t_abs, z0],
                [x1, t_abs, z0],
                [x1, t_abs, z1],
                [x0, t_abs, z1],
            ], dtype=np.float32)

        rect_local_abs = rect_local + lm
        rect_world = (rect_local_abs @ R.T) + COM  # row-vector

        base = len(vertices)
        vertices.extend([rect_world[0], rect_world[1], rect_world[2], rect_world[3]])
        indices.extend([
            (base + 0, base + 1),
            (base + 1, base + 2),
            (base + 2, base + 3),
            (base + 3, base + 0),
        ])
        return vertices, indices

    @ti.kernel
    def _set_static(self, count: int, idxs: ti.types.ndarray()):
        for k in range(count):
            p = idxs[k]
            self.ps.is_dynamic[p] = 0
            for d in ti.static(range(self.ps.dim)):
                self.ps.v[p][d] = 0.0
                self.ps.acceleration[p][d] = 0.0
            self.ps.m_inv[p] = 0.0
