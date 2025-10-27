import numpy as np
from typing import Optional


class SimulationCache:
    """Frame-level simulation cache with ring buffer and UI helpers.

    Stores and restores a consistent set of per-particle fields to prevent
    index misalignment after counting-sort. Also preserves solver time,
    animator time (provided by caller), emitter states, and numpy RNG state.
    """

    def __init__(self, particle_system, solver, max_steps=10):
        self.ps = particle_system
        self.solver = solver
        self.enabled = False
        self.max_steps = int(max_steps)
        self._cache = []
        self.last_rewind_steps = 0
        # Baseline snapshot (frame 0) for instant soft reset
        self._baseline = None

    def clear(self):
        self._cache.clear()
        self.last_rewind_steps = 0

    def set_enabled(self, value: bool):
        self.enabled = bool(value)
        if not self.enabled:
            self.clear()

    def set_max_steps(self, steps: int):
        self.max_steps = max(1, int(steps))
        # Trim immediately if needed
        while len(self._cache) > self.max_steps:
            self._cache.pop(0)


    def _snapshot(self, frame_cnt: int, anim_time: float | None = None):
        try:
            N = int(self.ps.particle_num[None])
            snap = {"frame": int(frame_cnt), "particle_num": N}
            particle_fields = [
                "object_id",
                "x", "x_old", "x0",
                "v", "v_adv",
                "acceleration",
                "m_V", "m", "m_inv", "m_V0",
                "density", "density0",
                "pressure", "divergence",
                "material", "color", "is_dynamic",
                "n",
                "cur2ori", "ori2cur",
            ]
            surface_fields = [
                "x_s", "x_0_s",
                "skinning_weight",
                "surface_neighbor_num", "surface_neighbor_idx",
                "surface_vertex_object_id",
            ]
            arr = {}
            for name in particle_fields:
                try:
                    arr[name] = getattr(self.ps, name).to_numpy()[:N].copy()
                except Exception:
                    pass
            for name in surface_fields:
                try:
                    arr[name] = getattr(self.ps, name).to_numpy().copy()
                except Exception:
                    pass
            try:
                if getattr(self.ps, "surface_face_num", 0) > 0:
                    arr["surface_faces"] = self.ps.surface_faces.to_numpy().copy()
                    arr["surface_face_object_id"] = self.ps.surface_face_object_id.to_numpy().copy()
            except Exception:
                pass
            snap["arr"] = arr

            # solver time
            try:
                snap["solver_time"] = float(getattr(self.solver, "time", 0.0))
            except Exception:
                snap["solver_time"] = 0.0

            # emitter states
            if hasattr(self.ps, "emitter_system") and self.ps.emitter_system:
                e_sys = self.ps.emitter_system
                e_states = []
                try:
                    for e in e_sys.emitters:
                        e_states.append({
                            "x": e.x.copy(),
                            "next_emit_time": float(e.next_emit_time),
                            "emit_counter": int(e.emit_counter),
                        })
                except Exception:
                    e_states = None
                snap["emitter_states"] = e_states
                snap["emitter_suppress"] = int(getattr(e_sys, "suppress_steps", 0))
            else:
                snap["emitter_states"] = None
                snap["emitter_suppress"] = 0

            snap["np_random_state"] = np.random.get_state()
            if anim_time is not None:
                snap["anim_time"] = float(anim_time)
            return snap
        except Exception:
            return None

    # -----------------------------
    # Baseline helpers
    # -----------------------------
    def snapshot_baseline(self, anim_time: float | None = None):
        """Capture a baseline snapshot after solver.initialize()."""
        self._baseline = self._snapshot(frame_cnt=0, anim_time=anim_time)

    def restore_baseline(self):
        """Restore from the baseline snapshot if available.

        Returns (ok, frame, anim_time)
        """
        if self._baseline is None:
            return False, None, None
        return self._restore(self._baseline)

    def push(self, frame_cnt: int, anim_time: float | None = None):
        if not self.enabled:
            return
        snap = self._snapshot(frame_cnt, anim_time)
        if snap is None:
            return
        self._cache.append(snap)
        if len(self._cache) > self.max_steps:
            self._cache.pop(0)

    def _find_snapshot_for_frame(self, target_frame: int):
        for s in reversed(self._cache):
            if s.get("frame", -1) == target_frame:
                return s
        return None

    def _restore(self, snap):
        if snap is None:
            return False, None, None
        try:
            N = int(snap["particle_num"])
            self.ps.particle_num[None] = N

            particle_fields = {
                "object_id", "x","x_old","x0","v","v_adv","acceleration",
                "m_V","m","m_inv","m_V0","density","density0",
                "pressure","divergence","material","color","is_dynamic",
                "n","cur2ori","ori2cur",
            }
            surface_like = {
                "x_s","x_0_s","skinning_weight",
                "surface_neighbor_num","surface_neighbor_idx",
                "surface_vertex_object_id",
                "surface_faces","surface_face_object_id",
            }

            if "arr" in snap and isinstance(snap["arr"], dict):
                for name, val in snap["arr"].items():
                    try:
                        buf = getattr(self.ps, name).to_numpy()
                        if name in surface_like:
                            # 표면/스키닝/face: 모양이 맞으면 전체 복원
                            if val.shape == buf.shape:
                                getattr(self.ps, name).from_numpy(val)
                            else:
                                # 모양 불일치 시 건너뜀(안전)
                                pass
                        else:
                            # 파티클 계열: 앞 N만 채움
                            buf[:N] = val
                            getattr(self.ps, name).from_numpy(buf)
                    except Exception:
                        pass

            # tail 비활성화
            M = int(self.ps.particle_max_num)
            if N < M:
                def _fill_tail(field_name, fill_value):
                    try:
                        buf = getattr(self.ps, field_name).to_numpy()
                        buf[N:] = fill_value
                        getattr(self.ps, field_name).from_numpy(buf)
                    except Exception:
                        pass
                _fill_tail("material", -1)
                _fill_tail("is_dynamic", 0)
                _fill_tail("object_id", -1)
                _fill_tail("m_V0", 0.0); _fill_tail("m_V", 0.0)
                _fill_tail("m", 0.0); _fill_tail("m_inv", 0.0)
                _fill_tail("density", 0.0); _fill_tail("density0", 0.0)
                _fill_tail("pressure", 0.0); _fill_tail("divergence", 0.0)
                _fill_tail("dfsph_factor", 0.0); _fill_tail("density_adv", 0.0)
                _fill_tail("x", 1000.0); _fill_tail("x_old", 1000.0); _fill_tail("x0", 1000.0)
                _fill_tail("v", 0.0); _fill_tail("v_adv", 0.0)
                _fill_tail("acceleration", 0.0)
                _fill_tail("n", 0.0)
                _fill_tail("color", 0)

            # solver time
            self.solver.time = float(snap.get("solver_time", 0.0))

            # emitters
            if hasattr(self.ps, "emitter_system") and self.ps.emitter_system and snap.get("emitter_states") is not None:
                e_sys = self.ps.emitter_system
                for e, st in zip(e_sys.emitters, snap["emitter_states"]):
                    e.x = st["x"].astype(np.float32)
                    e.next_emit_time = float(st["next_emit_time"])
                    e.emit_counter = int(st["emit_counter"])
                e_sys.suppress_steps = int(snap.get("emitter_suppress", 0))

            # RNG
            np.random.set_state(snap["np_random_state"])

            # 네이버 재구축
            if hasattr(self.solver, "ns"):
                self.solver.ns.is_cur2ori[None] = True
                self.solver.ns.broad_phase()
                self.solver.ns.narrow_phase(self.ps.x, self.ps.particle_neighbors_num, self.ps.particle_neighbors)
                self.solver.ns.narrow_phase_surface(self.ps.x_0_s, self.ps.x, self.ps.surface_neighbor_num, self.ps.surface_neighbor_idx)

            return True, int(snap.get("frame", 0)), float(snap.get("anim_time", None))
        except Exception:
            return False, None, None

    def rewind_one(self, current_frame: int):
        """Rewind to the latest cached snapshot before current_frame.

        Returns dict: {
            'restored': bool,
            'frame': int or None,
            'anim_time': float or None,
            'rewind_steps': int
        }
        """
        target = int(current_frame) - 1
        snap = self._find_snapshot_for_frame(target)
        if snap is None:
            for s in reversed(self._cache):
                if s.get("frame", -1) < int(current_frame):
                    snap = s
                    break
        if snap is None:
            self.last_rewind_steps = 0
            return {"restored": False, "frame": None, "anim_time": None, "rewind_steps": 0}

        diff = max(0, int(current_frame) - int(snap.get("frame", 0)))
        ok, frame_restored, anim_time_restored = self._restore(snap)
        self.last_rewind_steps = diff if ok else 0
        return {
            "restored": ok,
            "frame": frame_restored,
            "anim_time": anim_time_restored,
            "rewind_steps": self.last_rewind_steps,
        }

    def show_ui(self, gui, current_frame: int, pos=(0.4, 0.0), size=(0.3, 0.25)):
        """Render a small UI panel. Returns rewind result dict when rewound; otherwise {'restored': False}.
        """
        with gui.sub_window("Cache settings", pos[0], pos[1], size[0], size[1]) as w:
            prev_enabled = self.enabled
            self.enabled = w.checkbox("enable cache", self.enabled)
            self.max_steps = w.slider_int("cache steps", self.max_steps, 1, 200)
            if (not self.enabled) and prev_enabled:
                self.clear()
            gui.text(f"cached frames: {len(self._cache)}")
            gui.text(f"last rewind: {self.last_rewind_steps} frames")
            if self.enabled:
                if w.button("Rewind 1 frame"):
                    return self.rewind_one(current_frame)
                if w.button("Clear cache"):
                    self.clear()
        return {"restored": False}


