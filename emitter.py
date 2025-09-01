import math
import numpy as np
import taichi as ti

# type: 0=box, 1=circle (SPlisHSPlasH와 동일 의미)
class Emitter:
    def __init__(self, ps, width, height, pos, rotation, velocity, type=0,
                 start_time=0.0, end_time=1e9, object_id=0,
                 density=1000.0, color=(0, 150, 255),
                 jitter=0.25, spread_deg=5.0):
        self.ps = ps  # ParticleSystem
        self.width = int(width)
        self.height = int(height)
        self.x = np.array(pos, dtype=np.float32)
        self.x0 = self.x.copy()
        self.R = np.array(rotation, dtype=np.float32)  # 3x3
        self.v_emit = float(velocity)
        self.type = int(type)
        self.next_emit_time = 0.0
        self.emit_start_time = float(start_time)
        self.emit_end_time = float(end_time)
        self.emit_counter = 0
        self.object_id = int(object_id)
        self.density = float(density)
        self.color = np.array(color, dtype=np.int32)
        self.jitter = float(jitter)
        self.spread_deg = float(spread_deg)

    def reset(self):
        self.next_emit_time = self.emit_start_time
        self.emit_counter = 0
        # restore emitter origin exactly
        self.x = self.x0.copy().astype(np.float32)

    def get_size(self):
        radius = self.ps.particle_radius
        diam = 2.0 * radius
        support = self.ps.support_radius
        margin = diam
        if self.type == 0:
            return np.array([
                2.0 * support,
                self.height * diam + 2.0 * margin,
                self.width  * diam + 2.0 * margin
            ], dtype=np.float32)
        else:
            # cylinder: height along emitDir, radius from width
            h = 2.0 * support
            r = 0.5 * self.width * diam + margin
            return np.array([h, 2.0 * r, 2.0 * r], dtype=np.float32)

    def _grid_positions_box(self, t, dt):
        radius = self.ps.particle_radius
        diam = 2.0 * radius
        startX = -0.5 * (self.width  - 1) * diam
        startZ = -0.5 * (self.height - 1) * diam

        emit_dir = self.R[:, 0]
        axis_h = self.R[:, 1]
        axis_w = self.R[:, 2]
        # apply small angular spread to direction
        if self.spread_deg > 1e-6:
            deg = self.spread_deg
            # random small rotation around an axis perpendicular to emit_dir
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            if abs(np.dot(emit_dir, up)) > 0.95:
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            w = np.cross(up, emit_dir); w = w / (np.linalg.norm(w) + 1e-8)
            theta = (np.random.rand() * 2.0 - 1.0) * (deg * np.pi / 180.0)
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=np.float32)
            Rsmall = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
            emit_dir = (Rsmall @ emit_dir).astype(np.float32)
        emit_vel = self.v_emit * emit_dir

        dt_emit = (t - self.next_emit_time + dt)
        offset = self.x + dt_emit * emit_vel

        xs, ys, zs = [], [], []
        for i in range(self.width):
            for j in range(self.height):
                pos = (i*diam + startX) * axis_w + (j*diam + startZ) * axis_h + offset
                if self.jitter > 1e-12:
                    pos += (np.random.rand(3) - 0.5) * (self.jitter * diam)
                xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2])
        P = np.stack([xs, ys, zs], axis=1).astype(np.float32)

        V = np.tile(emit_vel[None, :], (P.shape[0], 1)).astype(np.float32)
        return P, V

    def _grid_positions_circle(self, t, dt):
        # SPlisHSPlasH Emitter::emitParticlesCircle
        radius = self.ps.particle_radius
        diam = 2.0 * radius
        startX = -0.5 * (self.width - 1) * diam
        startZ = -0.5 * (self.width - 1) * diam

        emit_dir = self.R[:, 0]
        axis_h = self.R[:, 1]
        axis_w = self.R[:, 2]
        emit_vel = self.v_emit * emit_dir

        dt_emit = (t - self.next_emit_time + dt)
        offset = self.x + dt_emit * emit_vel

        grid_r = 0.5 * self.width * diam
        r2 = grid_r * grid_r

        xs, ys, zs = [], [], []
        for i in range(self.width):
            for j in range(self.width):
                x = (i*diam + startX)
                y = (j*diam + startZ)
                if (x*x + y*y) <= r2:
                    pos = x * axis_w + y * axis_h + offset
                    xs.append(pos[0]); ys.append(pos[1]); zs.append(pos[2])
        P = np.stack([xs, ys, zs], axis=1).astype(np.float32)
        V = np.tile(emit_vel[None, :], (P.shape[0], 1)).astype(np.float32)
        return P, V

    def step(self, t, dt, reuse_indices):
        if (t < self.emit_start_time) or (t > self.emit_end_time):
            return 0, 0  # emitted, reused_this_step

        if t < self.next_emit_time:
            return 0, 0

        if self.type == 1:
            P, V = self._grid_positions_circle(t, dt)
        else:
            P, V = self._grid_positions_box(t, dt)

        n = P.shape[0]
        reused_now = min(len(reuse_indices), n)
        new_now = n - reused_now

        # Clamp new emission to remaining capacity when reuse is disabled or insufficient
        remain = int(self.ps.particle_max_num) - int(self.ps.particle_num[None])
        if remain < 0:
            remain = 0
        if new_now > remain:
            new_now = remain

        emitted = 0
        if reused_now > 0:
            idxs = np.array([reuse_indices.pop() for _ in range(reused_now)], dtype=np.int32)
            self.ps.set_emitted_on_indices(reused_now, idxs, P[:reused_now], V[:reused_now],
                                           self.object_id, self.density, self.color)

            emitted += reused_now

        if new_now > 0:
            self.ps.add_particles(
                object_id=self.object_id,
                new_particles_num=new_now,
                new_particles_positions=P[reused_now:],
                new_particles_velocity=V[reused_now:],
                new_particle_density=np.full(new_now, self.density, dtype=np.float32),
                new_particle_pressure=np.zeros(new_now, dtype=np.float32),
                new_particles_material=np.zeros(new_now, dtype=np.int32) + 1,  # 1=fluid
                new_particles_is_dynamic=np.ones(new_now, dtype=np.int32),     # dynamic
                new_particles_color=np.stack([
                    np.full(new_now, self.color[0], dtype=np.int32),
                    np.full(new_now, self.color[1], dtype=np.int32),
                    np.full(new_now, self.color[2], dtype=np.int32)
                ], axis=1)
            )
            emitted += new_now

        diam = 2.0 * self.ps.particle_radius
        if self.v_emit > 1e-12:
            self.next_emit_time += diam / self.v_emit
        else:
            self.next_emit_time = t + dt
        self.emit_counter += 1
        return emitted, reused_now


class EmitterSystem:
    def __init__(self, ps, max_reuse_per_step=50000):
        self.ps = ps
        self.emitters = []
        self.reuse_enabled = False
        self.box_min = np.array([-1., -1., -1.], dtype=np.float32)
        self.box_max = np.array([ 1.,  1.,  1.], dtype=np.float32)
        self.max_reuse_per_step = int(max_reuse_per_step)

        self.total_emitted = 0
        self.total_reused = 0
        self.suppress_steps = 0

    def reset(self):
        self.total_emitted = 0
        self.total_reused = 0
        for e in self.emitters:
            e.reset()
            # ensure emitter position/timebase exactly at start
            e.next_emit_time = e.emit_start_time
        # Suppress emission for the next solver step to avoid startup collisions
        self.suppress_steps = 1

    def add_emitter(self, *args, **kwargs):
        e = Emitter(self.ps, *args, **kwargs)
        self.emitters.append(e)
        return e

    def enable_reuse_particles(self, box_min=(-1, -1, -1), box_max=(1, 1, 1)):
        self.reuse_enabled = True
        self.box_min = np.array(box_min, dtype=np.float32)
        self.box_max = np.array(box_max, dtype=np.float32)

    def disable_reuse_particles(self):
        self.reuse_enabled = False

    def _collect_reuse_indices(self):
        if not self.reuse_enabled:
            return []

        N = int(self.ps.particle_num[None])
        if N <= 0:
            return []

        np_x = np.empty((N, 3), dtype=np.float32)
        self.ps.copy_to_numpy(np_x, self.ps.x)

        mask_min = np.zeros((N,), dtype=bool)
        mask_max = np.zeros((N,), dtype=bool)
        for d in range(self.ps.dim):
            mask_min |= (np_x[:, d] < self.box_min[d])
            mask_max |= (np_x[:, d] > self.box_max[d])
        out_mask = mask_min | mask_max

        np_mat = self.ps.material.to_numpy()[:N]
        fluid_mask = (np_mat == self.ps.material_fluid)

        idxs = np.nonzero(out_mask & fluid_mask)[0].astype(np.int32)
        return list(idxs.tolist())

    def step(self, t, dt):
        if len(self.emitters) == 0:
            return 0, 0

        if self.suppress_steps > 0:
            self.suppress_steps -= 1
            return 0, 0

        reuse_indices = self._collect_reuse_indices()

        emitted_sum = 0
        reused_sum = 0
        for e in self.emitters:
            em, ru = e.step(t, dt, reuse_indices)
            emitted_sum += em
            reused_sum += ru

        self.total_emitted += emitted_sum
        self.total_reused += reused_sum
        return emitted_sum, reused_sum
