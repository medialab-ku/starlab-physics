# randomizer.py
import taichi as ti

@ti.data_oriented
class ParticleRandomizer:
    def __init__(self, ps, ns):
        self.ps = ps
        self.ns = ns

        self.rand_unit = ti.Vector.field(3, dtype=float, shape=self.ps.particle_max_num)

        self.cube_center = ti.Vector.field(3, dtype=float, shape=self.ps.num_objects)
        self.cube_half   = ti.field(dtype=float, shape=self.ps.num_objects)
        self.cube_min    = ti.Vector.field(3, dtype=float, shape=self.ps.num_objects)
        self.cube_max    = ti.Vector.field(3, dtype=float, shape=self.ps.num_objects)
        self.cube_count  = ti.field(dtype=int,   shape=self.ps.num_objects)

        self.t = 0.0
        self.seed = 1337
        self.active = False

    @ti.kernel
    def compute_cube(self, expand: float):
        for o in range(self.ps.num_objects):
            self.cube_center[o] = ti.math.vec3(0.0)
            self.cube_half[o]   = 0.0
            self.cube_min[o]    = ti.math.vec3(1e30)
            self.cube_max[o]    = ti.math.vec3(-1e30)
            self.cube_count[o]  = 0

        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] == self.ps.material_solid:
                o = self.ps.object_id[p]
                x0 = self.ps.x0[p]
                self.cube_min[o][0] = ti.min(self.cube_min[o][0], x0[0])
                self.cube_min[o][1] = ti.min(self.cube_min[o][1], x0[1])
                self.cube_min[o][2] = ti.min(self.cube_min[o][2], x0[2])
                self.cube_max[o][0] = ti.max(self.cube_max[o][0], x0[0])
                self.cube_max[o][1] = ti.max(self.cube_max[o][1], x0[1])
                self.cube_max[o][2] = ti.max(self.cube_max[o][2], x0[2])
                ti.atomic_add(self.cube_count[o], 1)
                self.cube_center[o] += x0

        for o in range(self.ps.num_objects):
            c = ti.max(1, self.cube_count[o])
            self.cube_center[o] /= float(c)

            ext = self.cube_max[o] - self.cube_min[o]
            max_len = ti.max(ext[0], ti.max(ext[1], ext[2]))
            base = ti.max(max_len, self.ps.support_radius * 2.0)
            s_raw = 0.5 * base * expand

            pad = self.ps.padding
            dx = ti.min(self.ps.domain_size[0] - pad - self.cube_center[o][0], self.cube_center[o][0] - pad)
            dy = ti.min(self.ps.domain_size[1] - pad - self.cube_center[o][1], self.cube_center[o][1] - pad)
            dz = ti.min(self.ps.domain_size[2] - pad - self.cube_center[o][2], self.cube_center[o][2] - pad)
            s_dom = ti.max(0.0, ti.min(s_raw, ti.min(dx, ti.min(dy, dz))))

            self.cube_half[o] = s_dom

    @ti.kernel
    def build_rand_dirs(self, seed: int):
        seed_f = ti.cast(seed, ti.f32)
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] == self.ps.material_solid:
                k  = seed_f + ti.cast(p, ti.f32) * 0.123457
                r0 = ti.sin(k + 0.0)   * 43758.5453123
                r1 = ti.sin(k + 17.23) * 43758.5453123
                r2 = ti.sin(k + 93.57) * 43758.5453123
                rx = (r0 - ti.floor(r0)) * 2.0 - 1.0
                ry = (r1 - ti.floor(r1)) * 2.0 - 1.0
                rz = (r2 - ti.floor(r2)) * 2.0 - 1.0
                self.rand_unit[p] = ti.math.vec3(rx, ry, rz)  # [-1,1]^3

    @ti.kernel
    def update_positions(self, t: float):
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] == self.ps.material_solid:
                o = self.ps.object_id[p]
                c = self.cube_center[o]
                s = self.cube_half[o] * t
                u = self.rand_unit[p]
                x = c + u * s
                x[0] = ti.min(self.ps.domain_size[0] - self.ps.padding, ti.max(self.ps.padding, x[0]))
                x[1] = ti.min(self.ps.domain_size[1] - self.ps.padding, ti.max(self.ps.padding, x[1]))
                x[2] = ti.min(self.ps.domain_size[2] - self.ps.padding, ti.max(self.ps.padding, x[2]))
                self.ps.x[p] = x

    def begin(self, expand: float, seed: int = 1337):
        self.compute_cube(expand)
        self.seed = int(seed)
        self.t = 0.0
        self.build_rand_dirs(self.seed)
        self.active = True

    def step(self, alpha: float = 0.05):
        if not self.active:
            return
        self.t = min(1.0, self.t + alpha)
        self.update_positions(self.t)

        self.ns.broad_phase()
        self.ns.narrow_phase(self.ps.x)
        self.ns.enforce_boundary_3D()

        if self.t >= 1.0:
            self.active = False