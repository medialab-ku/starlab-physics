import taichi as ti
import numpy as np


@ti.data_oriented
class AnimationSystem:
    def __init__(self, particle_system, config) -> None:
        self.ps = particle_system
        self.cfg = config
        self.time = 0.0
        self.anims = self._parse_animations()
        # Toggle-gated auto animations
        self._enabled_auto_anim_obj_ids = set()
        # Per-object animation start time offset when enabled mid-sim
        self._enabled_anim_t0 = {}
        # Manual control state (only for non-auto anims)
        self._translate_offset = {}
        self._rotate_angle = {}
        for a in self.anims:
            if bool(a.get("auto", False)):
                continue
            key = (int(a["obj"]), int(a["axis"]))
            if int(a["type"]) == 0:
                self._translate_offset[key] = 0.0
            else:
                self._rotate_angle[key] = 0.0

    def has_animations(self) -> bool:
        return len(self.anims) > 0

    def reset(self):
        self.time = 0.0
        self.anims = self._parse_animations()
        self._enabled_auto_anim_obj_ids.clear()
        self._enabled_anim_t0.clear()
        # Rebuild manual state
        self._translate_offset = {}
        self._rotate_angle = {}
        for a in self.anims:
            if bool(a.get("auto", False)):
                continue
            key = (int(a["obj"]), int(a["axis"]))
            if int(a["type"]) == 0:
                self._translate_offset[key] = 0.0
            else:
                self._rotate_angle[key] = 0.0

    def reset_manual(self):
        # Zero all manual offsets/angles
        for k in list(self._translate_offset.keys()):
            self._translate_offset[k] = 0.0
        for k in list(self._rotate_angle.keys()):
            self._rotate_angle[k] = 0.0

    def _parse_axis(self, ax):
        if isinstance(ax, str):
            a = ax.lower()
            return 0 if a == "x" else (1 if a == "y" else 2)
        return int(ax)

    def _parse_animations(self):
        out = []
        try:
            # Rigid bodies
            for rb in (self.cfg.get_rigid_bodies() or []):
                if not isinstance(rb, dict):
                    continue
                anim = rb.get("animation")
                if not isinstance(anim, dict):
                    continue
                obj = int(rb.get("objectId", -1))
                typ = str(anim.get("type", "oscillate")).lower()
                axis = self._parse_axis(anim.get("axis", "z"))
                axis_vec = anim.get("axisVec", None)
                pivot = anim.get("pivot", None)
                is_toggled = bool(rb.get("isToggled", False))
                amp = float(anim.get("amplitude", 0.0))
                period = float(anim.get("period", 1.0))
                phase = float(anim.get("phase", 0.0))
                is_auto = bool(anim.get("auto", False))
                omega = (2.0 * np.pi) / max(1e-6, period)
                if typ == "oscillate":
                    base = 0.5 * amp * (1.0 - np.cos(phase))
                    out.append({"obj": obj, "type": 0, "axis": axis, "amp": amp, "omega": omega, "phase": phase, "base": base, "auto": is_auto, "isToggled": is_toggled})
                else:
                    base = phase
                    out.append({"obj": obj, "type": 1, "axis": axis, "axisVec": axis_vec, "pivot": pivot, "amp": 0.0, "omega": omega, "phase": phase, "base": base, "auto": is_auto, "isToggled": is_toggled})

            # Rigid blocks (optional support)
            for rb in (self.cfg.get_rigid_blocks() or []):
                if not isinstance(rb, dict):
                    continue
                anim = rb.get("animation")
                if not isinstance(anim, dict):
                    continue
                obj = int(rb.get("objectId", -1))
                typ = str(anim.get("type", "oscillate")).lower()
                axis = self._parse_axis(anim.get("axis", "z"))
                axis_vec = anim.get("axisVec", None)
                pivot = anim.get("pivot", None)
                is_toggled = bool(rb.get("isToggled", False))
                amp = float(anim.get("amplitude", 0.0))
                period = float(anim.get("period", 1.0))
                phase = float(anim.get("phase", 0.0))
                is_auto = bool(anim.get("auto", False))
                omega = (2.0 * np.pi) / max(1e-6, period)
                if typ == "oscillate":
                    base = 0.5 * amp * (1.0 - np.cos(phase))
                    out.append({"obj": obj, "type": 0, "axis": axis, "amp": amp, "omega": omega, "phase": phase, "base": base, "auto": is_auto, "isToggled": is_toggled})
                else:
                    base = phase
                    out.append({"obj": obj, "type": 1, "axis": axis, "axisVec": axis_vec, "pivot": pivot, "amp": 0.0, "omega": omega, "phase": phase, "base": base, "auto": is_auto, "isToggled": is_toggled})
        except Exception:
            pass
        return out

    # -----------------------------
    # Manual-control kernels
    # -----------------------------
    @ti.kernel
    def _apply_translate_manual(self, obj_id: int, axis: int, offset: float,
                                x: ti.template(), x0: ti.template(), object_id: ti.template()):
        for p_i in ti.grouped(x):
            if object_id[p_i] == obj_id:
                disp = ti.math.vec3(0.0)
                disp[axis] = offset
                x[p_i] = x0[p_i] + disp

    @ti.kernel
    def _apply_rotate_manual(self, obj_id: int, axis: int, angle: float,
                             x: ti.template(), x0: ti.template(), object_id: ti.template(), rest_cm: ti.template()):
        c = ti.cos(angle)
        s = ti.sin(angle)
        R = ti.Matrix.zero(float, 3, 3)
        if axis == 0:
            R[0, 0] = 1.0; R[0, 1] = 0.0; R[0, 2] = 0.0
            R[1, 0] = 0.0; R[1, 1] = c;   R[1, 2] = -s
            R[2, 0] = 0.0; R[2, 1] = s;   R[2, 2] = c
        elif axis == 1:
            R[0, 0] = c;   R[0, 1] = 0.0; R[0, 2] = s
            R[1, 0] = 0.0; R[1, 1] = 1.0; R[1, 2] = 0.0
            R[2, 0] = -s;  R[2, 1] = 0.0; R[2, 2] = c
        else:
            R[0, 0] = c;   R[0, 1] = -s;  R[0, 2] = 0.0
            R[1, 0] = s;   R[1, 1] = c;   R[1, 2] = 0.0
            R[2, 0] = 0.0; R[2, 1] = 0.0; R[2, 2] = 1.0

        com0 = rest_cm[obj_id]
        for p_i in ti.grouped(x):
            if object_id[p_i] == obj_id:
                local = x0[p_i] - com0
                x[p_i] = com0 + R @ local

    # -----------------------------
    # Manual-control APIs
    # -----------------------------
    def nudge_all_translate(self, delta: float):
        for a in self.anims:
            if bool(a.get("auto", False)):
                continue
            if int(a["type"]) == 0:
                key = (int(a["obj"]), int(a["axis"]))
                self._translate_offset[key] = float(self._translate_offset.get(key, 0.0)) + float(delta)

    def nudge_all_rotate(self, delta_angle: float):
        for a in self.anims:
            if bool(a.get("auto", False)):
                continue
            if int(a["type"]) != 0:
                key = (int(a["obj"]), int(a["axis"]))
                self._rotate_angle[key] = float(self._rotate_angle.get(key, 0.0)) + float(delta_angle)

    def nudge_translate_axis(self, axis: int, delta: float):
        # Nudge only animations that are translate type and match axis
        d = float(delta)
        for a in self.anims:
            if bool(a.get("auto", False)):
                continue
            if int(a["type"]) == 0 and int(a["axis"]) == int(axis):
                key = (int(a["obj"]), int(axis))
                self._translate_offset[key] = float(self._translate_offset.get(key, 0.0)) + d

    def nudge_rotate_axis(self, axis: int, delta_angle: float):
        # Nudge only animations that are rotate type and match axis
        d = float(delta_angle)
        for a in self.anims:
            if bool(a.get("auto", False)):
                continue
            if int(a["type"]) != 0 and int(a["axis"]) == int(axis):
                key = (int(a["obj"]), int(axis))
                self._rotate_angle[key] = float(self._rotate_angle.get(key, 0.0)) + d

    def apply_manual(self):
        # Apply current manual states to MANUAL animations only
        for a in self.anims:
            if bool(a.get("auto", False)):
                continue
            obj = int(a["obj"])
            axis = int(a["axis"])
            if int(a["type"]) == 0:
                offset = float(self._translate_offset.get((obj, axis), 0.0))
                self._apply_translate_manual(obj, axis, offset, self.ps.x, self.ps.x_0, self.ps.object_id)
            else:
                angle = float(self._rotate_angle.get((obj, axis), 0.0))
                self._apply_rotate_manual(obj, axis, angle, self.ps.x, self.ps.x_0, self.ps.object_id, self.ps.rigid_rest_cm)

    def apply(self, t_now: float):
        # Apply AUTO animations only
        if len(self.anims) == 0:
            return
        for a in self.anims:
            if not bool(a.get("auto", False)):
                continue
            # Gate auto animation if this object is toggled and not yet enabled
            if bool(a.get("isToggled", False)) and (int(a.get("obj", -1)) not in self._enabled_auto_anim_obj_ids):
                continue
            # Shift time so animation starts at 0 when toggled mid-sim
            t_eval = float(t_now)
            try:
                t0 = float(self._enabled_anim_t0.get(int(a.get("obj", -1)), 0.0))
                t_eval = float(max(0.0, t_now - t0))
            except Exception:
                t_eval = float(t_now)
            if int(a["type"]) == 0:
                self._animate_translate_axis(int(a["obj"]), int(a["axis"]), float(a["amp"]), float(a["omega"]), float(a["phase"]), float(a["base"]), float(t_eval),
                                             self.ps.x, self.ps.x_0, self.ps.object_id)
            else:
                # Prefer arbitrary axis and pivot if provided
                axis_vec = a.get("axisVec", None)
                pivot = a.get("pivot", None)
                if isinstance(axis_vec, list) and len(axis_vec) == 3:
                    ax, ay, az = float(axis_vec[0]), float(axis_vec[1]), float(axis_vec[2])
                else:
                    # Fallback to principal axes
                    ax = 1.0 if int(a["axis"]) == 0 else 0.0
                    ay = 1.0 if int(a["axis"]) == 1 else 0.0
                    az = 1.0 if int(a["axis"]) == 2 else 0.0
                if isinstance(pivot, list) and len(pivot) == 3:
                    px, py, pz = float(pivot[0]), float(pivot[1]), float(pivot[2])
                    self._animate_rotate_axis_with_pivot(int(a["obj"]), ax, ay, az, px, py, pz, float(a["omega"]), float(a["phase"]), float(a["base"]), float(t_eval),
                                                         self.ps.x, self.ps.x_0, self.ps.object_id)
                else:
                    self._animate_rotate_axis(int(a["obj"]), ax, ay, az, float(a["omega"]), float(a["phase"]), float(a["base"]), float(t_eval),
                                              self.ps.x, self.ps.x_0, self.ps.object_id, self.ps.rigid_rest_cm)

    def has_auto(self) -> bool:
        for a in self.anims:
            if bool(a.get("auto", False)):
                return True
        return False

    def has_animation_for(self, obj_id: int) -> bool:
        for a in self.anims:
            try:
                if int(a.get("obj", -1)) == int(obj_id):
                    return True
            except Exception:
                pass
        return False

    def enable_animation_for(self, obj_id: int):
        try:
            self._enabled_auto_anim_obj_ids.add(int(obj_id))
        except Exception:
            pass

    def reset_toggle_activation(self):
        self._enabled_auto_anim_obj_ids.clear()
        self._enabled_anim_t0.clear()

    def has_enabled_auto(self) -> bool:
        try:
            return len(self._enabled_auto_anim_obj_ids) > 0
        except Exception:
            return False

    def enable_animation_for_with_time(self, obj_id: int, current_time: float):
        try:
            oid = int(obj_id)
            self._enabled_auto_anim_obj_ids.add(oid)
            self._enabled_anim_t0[oid] = float(current_time)
        except Exception:
            pass

    def has_manual(self) -> bool:
        for a in self.anims:
            if not bool(a.get("auto", False)):
                return True
        return False

    @ti.kernel
    def _animate_translate_axis(self, obj_id: int, axis: int, amplitude: float, omega: float, phase: float, base_offset: float, t: float,
                                x: ti.template(), x0: ti.template(), object_id: ti.template()):
        # displacement within [0, amplitude], start at 0 on t=0 by subtracting baseline
        disp_scalar = 0.5 * amplitude * (1.0 - ti.cos(omega * t + phase)) - base_offset
        for p_i in ti.grouped(x):
            if object_id[p_i] == obj_id:
                disp = ti.math.vec3(0.0)
                disp[axis] = disp_scalar
                x[p_i] = x0[p_i] + disp

    @ti.kernel
    def _animate_rotate_axis(self, obj_id: int, ax: float, ay: float, az: float, omega: float, phase: float, base_angle: float, t: float,
                             x: ti.template(), x0: ti.template(), object_id: ti.template(), rest_cm: ti.template()):
        angle = (omega * t + phase) - base_angle
        # Normalize axis
        nx = ax
        ny = ay
        nz = az
        nrm = ti.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= nrm; ny /= nrm; nz /= nrm
        c = ti.cos(angle)
        s = ti.sin(angle)
        one_c = 1.0 - c
        # Rodrigues' rotation matrix
        R = ti.Matrix.zero(float, 3, 3)
        R[0, 0] = c + nx * nx * one_c
        R[0, 1] = nx * ny * one_c - nz * s
        R[0, 2] = nx * nz * one_c + ny * s
        R[1, 0] = ny * nx * one_c + nz * s
        R[1, 1] = c + ny * ny * one_c
        R[1, 2] = ny * nz * one_c - nx * s
        R[2, 0] = nz * nx * one_c - ny * s
        R[2, 1] = nz * ny * one_c + nx * s
        R[2, 2] = c + nz * nz * one_c

        pivot = rest_cm[obj_id]
        for p_i in ti.grouped(x):
            if object_id[p_i] == obj_id:
                local = x0[p_i] - pivot
                x[p_i] = pivot + R @ local

    @ti.kernel
    def _animate_rotate_axis_with_pivot(self, obj_id: int, ax: float, ay: float, az: float, px: float, py: float, pz: float, omega: float, phase: float, base_angle: float, t: float,
                                        x: ti.template(), x0: ti.template(), object_id: ti.template()):
        angle = (omega * t + phase) - base_angle
        # Normalize axis
        nx = ax
        ny = ay
        nz = az
        nrm = ti.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
        nx /= nrm; ny /= nrm; nz /= nrm
        c = ti.cos(angle)
        s = ti.sin(angle)
        one_c = 1.0 - c
        # Rodrigues' rotation matrix
        R = ti.Matrix.zero(float, 3, 3)
        R[0, 0] = c + nx * nx * one_c
        R[0, 1] = nx * ny * one_c - nz * s
        R[0, 2] = nx * nz * one_c + ny * s
        R[1, 0] = ny * nx * one_c + nz * s
        R[1, 1] = c + ny * ny * one_c
        R[1, 2] = ny * nz * one_c - nx * s
        R[2, 0] = nz * nx * one_c - ny * s
        R[2, 1] = nz * ny * one_c + nx * s
        R[2, 2] = c + nz * nz * one_c

        pivot = ti.Vector([px, py, pz])
        for p_i in ti.grouped(x):
            if object_id[p_i] == obj_id:
                local = x0[p_i] - pivot
                x[p_i] = pivot + R @ local

