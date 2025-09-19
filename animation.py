import taichi as ti
import numpy as np


@ti.data_oriented
class AnimationSystem:
    def __init__(self, particle_system, config) -> None:
        self.ps = particle_system
        self.cfg = config
        self.time = 0.0
        self.anims = self._parse_animations()
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
                amp = float(anim.get("amplitude", 0.0))
                period = float(anim.get("period", 1.0))
                phase = float(anim.get("phase", 0.0))
                is_auto = bool(anim.get("auto", False))
                omega = (2.0 * np.pi) / max(1e-6, period)
                if typ == "oscillate":
                    base = 0.5 * amp * (1.0 - np.cos(phase))
                    out.append({"obj": obj, "type": 0, "axis": axis, "amp": amp, "omega": omega, "phase": phase, "base": base, "auto": is_auto})
                else:
                    base = phase
                    out.append({"obj": obj, "type": 1, "axis": axis, "amp": 0.0, "omega": omega, "phase": phase, "base": base, "auto": is_auto})

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
                amp = float(anim.get("amplitude", 0.0))
                period = float(anim.get("period", 1.0))
                phase = float(anim.get("phase", 0.0))
                is_auto = bool(anim.get("auto", False))
                omega = (2.0 * np.pi) / max(1e-6, period)
                if typ == "oscillate":
                    base = 0.5 * amp * (1.0 - np.cos(phase))
                    out.append({"obj": obj, "type": 0, "axis": axis, "amp": amp, "omega": omega, "phase": phase, "base": base, "auto": is_auto})
                else:
                    base = phase
                    out.append({"obj": obj, "type": 1, "axis": axis, "amp": 0.0, "omega": omega, "phase": phase, "base": base, "auto": is_auto})
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
            if int(a["type"]) == 0:
                self._animate_translate_axis(int(a["obj"]), int(a["axis"]), float(a["amp"]), float(a["omega"]), float(a["phase"]), float(a["base"]), float(t_now),
                                             self.ps.x, self.ps.x_0, self.ps.object_id)
            else:
                self._animate_rotate_axis(int(a["obj"]), int(a["axis"]), float(a["omega"]), float(a["phase"]), float(a["base"]), float(t_now),
                                          self.ps.x, self.ps.x_0, self.ps.object_id, self.ps.rigid_rest_cm)

    def has_auto(self) -> bool:
        for a in self.anims:
            if bool(a.get("auto", False)):
                return True
        return False

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
    def _animate_rotate_axis(self, obj_id: int, axis: int, omega: float, phase: float, base_angle: float, t: float,
                             x: ti.template(), x0: ti.template(), object_id: ti.template(), rest_cm: ti.template()):
        # start at rest orientation on t=0 by subtracting initial phase
        angle = (omega * t + phase) - base_angle
        c = ti.cos(angle)
        s = ti.sin(angle)
        # Build rotation matrix around principal axis
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

    def apply(self, t_now: float):
        if len(self.anims) == 0:
            return
        for a in self.anims:
            if a["type"] == 0:
                self._animate_translate_axis(int(a["obj"]), int(a["axis"]), float(a["amp"]), float(a["omega"]), float(a["phase"]), float(a["base"]), float(t_now),
                                             self.ps.x, self.ps.x_0, self.ps.object_id)
            else:
                self._animate_rotate_axis(int(a["obj"]), int(a["axis"]), float(a["omega"]), float(a["phase"]), float(a["base"]), float(t_now),
                                          self.ps.x, self.ps.x_0, self.ps.object_id, self.ps.rigid_rest_cm)


