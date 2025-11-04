import taichi as ti

@ti.data_oriented
class CubeRandomizer:
    def __init__(self, ps):
        self.ps = ps
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=ps.num_objects)
        self.half   = ti.field(dtype=ti.f32, shape=ps.num_objects)

    @ti.func
    def rand3(self, pid: ti.i32, seed: ti.i32):
        k  = ti.cast(seed, ti.f32) + ti.cast(pid, ti.f32) * 0.123457
        r0 = ti.sin(k + 0.00) * 43758.5453123
        r1 = ti.sin(k + 17.23) * 43758.5453123
        r2 = ti.sin(k + 93.57) * 43758.5453123
        rx = (r0 - ti.floor(r0)) * 2.0 - 1.0
        ry = (r1 - ti.floor(r1)) * 2.0 - 1.0
        rz = (r2 - ti.floor(r2)) * 2.0 - 1.0
        return ti.math.vec3(rx, ry, rz)
    @ti.kernel
    def compute_cubes(self, expand: ti.f32):
        for o in range(self.ps.num_objects):
            mn = ti.math.vec3(1e30)
            mx = ti.math.vec3(-1e30)
            c  = ti.math.vec3(0.0)
            n  = 0
            for p in range(self.ps.particle_num[None]):
                if self.ps.material[p] == self.ps.material_solid and self.ps.object_id[p] == o:
                    x0 = self.ps.x0[p]
                    mn[0] = ti.min(mn[0], x0[0]); mn[1] = ti.min(mn[1], x0[1]); mn[2] = ti.min(mn[2], x0[2])
                    mx[0] = ti.max(mx[0], x0[0]); mx[1] = ti.max(mx[1], x0[1]); mx[2] = ti.max(mx[2], x0[2])
                    c += x0
                    n += 1
            self.center[o] = c / ti.max(1, n)
            ext = mx - mn
            max_len = ti.max(ext[0], ti.max(ext[1], ext[2]))
            self.half[o] = 0.5 * max_len * expand
    @ti.kernel
    def randomize_into_cubes(self, seed: ti.i32):
        for p in range(self.ps.particle_num[None]):
            if self.ps.material[p] == self.ps.material_solid:
                o = self.ps.object_id[p]
                u = self.rand3(p, seed)            # [-1,1]^3
                self.ps.x[p] = self.center[o] + self.half[o] * u

    def run(self, expand: float = 1.0, seed: int = 1337):
        self.compute_cubes(expand)
        self.randomize_into_cubes(seed)
