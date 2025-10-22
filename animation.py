import taichi as ti
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class AnimationTrack:
    def __init__(self, object_id: int, target_type: str, pivot: str,
                 interpolation: str, channels: Dict[str, List[Dict[str, Any]]],
                 indices: np.ndarray, pivot_rest: np.ndarray):
        self.object_id = int(object_id)
        self.target_type = str(target_type)
        self.pivot = str(pivot)
        self.interp = str(interpolation or "linear")
        self.channels = channels or {}
        # 주의: 여기의 indices는 “원본 인덱스(original index)”로 취급 (pinning 직후, sort 이전)
        self.indices_np = indices.astype(np.int32)
        self.pivot_rest = pivot_rest.astype(np.float32)

@ti.data_oriented
class AnimationEngine:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.tracks: List[AnimationTrack] = []
        # 트랙별 인덱스 필드(원본 인덱스 저장), 크기
        self.track_idx_fields: List[ti.field] = []
        self.track_sizes: List[int] = []

    def build(self, scene_data: Dict, pin_geom: Optional[Dict] = None) -> None:
        self.tracks = []
        self.track_idx_fields = []
        self.track_sizes = []

        ranges_meta = (pin_geom or {}).get("ranges_meta", []) if pin_geom else []
        by_obj_ranges: Dict[int, Dict[str, Dict]] = {}
        for m in ranges_meta:
            oid = int(m["objectId"]); rid = str(m["rangeId"])
            by_obj_ranges.setdefault(oid, {})[rid] = m

        solids = scene_data.get("solid_bodies", [])
        obj_map = scene_data.get("object_map", {})

        N = int(self.ps.particle_num[None])
        object_id_np = self.ps.object_id.to_numpy()[:N]
        material_np = self.ps.material.to_numpy()[:N]

        for solid in solids:
            obj_id = int(solid["objectId"])
            if obj_id not in obj_map:
                continue
            cfg = obj_map[obj_id]

            anim_cfg = dict(cfg.get("Animation", {}) or {})  # 소문자 animation
            tracks_cfg = list(anim_cfg.get("tracks", []) or [])
            if not tracks_cfg:
                continue

            obj_rest_COM = np.array(cfg.get("restCenterOfMass", [0,0,0]), dtype=np.float32)
            obj_mask = (object_id_np == obj_id) & (material_np == self.ps.material_solid)
            obj_indices = np.nonzero(obj_mask)[0].astype(np.int32)  # 이 시점의 인덱스=원본 인덱스

            for tcfg in tracks_cfg:
                tgt = dict(tcfg.get("target", {}) or {})
                t_type = str(tgt.get("type", "rigid_body"))
                pivot = str(tcfg.get("pivot", "object_restCOM"))
                interp = str(tcfg.get("interpolation", "linear"))
                channels = dict(tcfg.get("channels", {}) or {})

                if t_type not in ("rigid_range", "rigid_body", "solid_body"):
                    continue

                if t_type == "rigid_range":
                    rid = str(tgt.get("rangeId", ""))
                    meta = (by_obj_ranges.get(obj_id, {}) or {}).get(rid)
                    if meta is None:
                        continue
                    idxs = meta["indices"].astype(np.int32)      # 원본 인덱스
                    pivot_rest = np.array(meta["restCOM"], dtype=np.float32) if pivot == "range_restCOM" else obj_rest_COM
                else:
                    idxs = obj_indices                           # 원본 인덱스(오브젝트 전체)
                    pivot_rest = obj_rest_COM

                if idxs.size == 0:
                    continue

                track = AnimationTrack(obj_id, t_type, pivot, interp, channels, idxs, pivot_rest)
                self.tracks.append(track)

                # 트랙별 인덱스 필드(한번만 업로드 → 이후 커널에서 ori2cur로 매핑)
                idx_field = ti.field(dtype=int, shape=idxs.size)
                idx_field.from_numpy(idxs)
                self.track_idx_fields.append(idx_field)
                self.track_sizes.append(int(idxs.size))

    @ti.kernel
    def _apply_track_kernel(self,
                            idxs: ti.template(),
                            count: int,
                            pivot_x: float, pivot_y: float, pivot_z: float,
                            R00: float, R01: float, R02: float,
                            R10: float, R11: float, R12: float,
                            R20: float, R21: float, R22: float,
                            Sx: float, Sy: float, Sz: float,
                            Tx: float, Ty: float, Tz: float,
                            dt: float, update_v: ti.i32):
        for k in range(count):
            orig = idxs[k]
            cur = self.ps.ori2cur[orig]  # 원본→현재 인덱스 매핑(NeighborSearch 정렬 대응)
            # 안전 범위 체크는 생략 가능
            X0 = self.ps.x0[cur]
            # (X0 - pivot) ⊙ S
            d0 = (X0[0] - pivot_x) * Sx
            d1 = (X0[1] - pivot_y) * Sy
            d2 = (X0[2] - pivot_z) * Sz
            # R * d
            rx = R00 * d0 + R01 * d1 + R02 * d2
            ry = R10 * d0 + R11 * d1 + R12 * d2
            rz = R20 * d0 + R21 * d1 + R22 * d2
            nx = pivot_x + Tx + rx
            ny = pivot_y + Ty + ry
            nz = pivot_z + Tz + rz
            if update_v != 0 and dt > 0.0:
                vx = (nx - self.ps.x[cur][0]) / dt
                vy = (ny - self.ps.x[cur][1]) / dt
                vz = (nz - self.ps.x[cur][2]) / dt
                self.ps.v[cur] = ti.Vector([vx, vy, vz])
            self.ps.x[cur] = ti.Vector([nx, ny, nz])

    def apply(self, t: float, dt: Optional[float] = None) -> None:
        if not self.tracks:
            return
        upd_v = 1 if (dt is not None and dt > 0.0) else 0

        for i, track in enumerate(self.tracks):
            T = self.sample_channel(track.channels.get("translation"), t, default=np.array([0.0,0.0,0.0], dtype=np.float32))
            R_euler = self.sample_channel(track.channels.get("rotation_euler_deg"), t, default=np.array([0.0,0.0,0.0], dtype=np.float32))
            S = self.sample_channel(track.channels.get("scale"), t, default=np.array([1.0,1.0,1.0], dtype=np.float32))
            Rm = self.euler_deg_xyz_to_matrix(R_euler)
            piv = track.pivot_rest

            idxs = self.track_idx_fields[i]
            count = self.track_sizes[i]
            self._apply_track_kernel(
                idxs, count,
                float(piv[0]), float(piv[1]), float(piv[2]),
                float(Rm[0,0]), float(Rm[0,1]), float(Rm[0,2]),
                float(Rm[1,0]), float(Rm[1,1]), float(Rm[1,2]),
                float(Rm[2,0]), float(Rm[2,1]), float(Rm[2,2]),
                float(S[0]), float(S[1]), float(S[2]),
                float(T[0]), float(T[1]), float(T[2]),
                float(dt or 0.0), int(upd_v)
            )

    @staticmethod
    def sample_channel(keyframes: Optional[List[Dict[str, Any]]], t: float, default: np.ndarray) -> np.ndarray:
        if not keyframes:
            return default.astype(np.float32)
        if len(keyframes) == 1:
            v = np.array(keyframes[0].get("value", default), dtype=np.float32)
            return v

        # assume sorted by time; if not, sort
        kfs = sorted(keyframes, key=lambda k: float(k.get("t", 0.0)))
        if t <= float(kfs[0]["t"]):
            return np.array(kfs[0]["value"], dtype=np.float32)
        if t >= float(kfs[-1]["t"]):
            return np.array(kfs[-1]["value"], dtype=np.float32)

        # linear interpolation
        for i in range(len(kfs) - 1):
            t0 = float(kfs[i]["t"]); t1 = float(kfs[i + 1]["t"])
            if t0 <= t <= t1:
                v0 = np.array(kfs[i]["value"], dtype=np.float32)
                v1 = np.array(kfs[i + 1]["value"], dtype=np.float32)
                w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return (1.0 - w) * v0 + w * v1

        return np.array(kfs[-1]["value"], dtype=np.float32)

    @staticmethod
    def euler_deg_xyz_to_matrix(euler_deg: np.ndarray) -> np.ndarray:
        # XYZ intrinsic order
        ex, ey, ez = np.deg2rad(euler_deg.astype(np.float32))
        cx, sx = np.cos(ex), np.sin(ex)
        cy, sy = np.cos(ey), np.sin(ey)
        cz, sz = np.cos(ez), np.sin(ez)

        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0],
                       [sz, cz, 0],
                       [0, 0, 1]], dtype=np.float32)
        # R = Rz * Ry * Rx (intrinsic XYZ)
        return (Rz @ Ry @ Rx).astype(np.float32)