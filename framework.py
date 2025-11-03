import taichi as ti
import time
@ti.data_oriented
class Framework:
    def __init__(self, particle_system, neighbour_search, pressure, viscosity, surface_tension, elasticity):
        # super().__init__(particle_system)

        self.ps = particle_system
        self.ns = neighbour_search
        self.pressure = pressure
        self.viscosity = viscosity
        self.surface_tension = surface_tension
        self.elasticity = elasticity

        self.dt = self.ps.cfg.get_cfg("timeStepSize")
        self.g = self.ps.cfg.get_cfg("gravitation")  # Gravity
        self.viscosity_coeff = 0.01            # viscosity
        self.surface_tension_coeff = 0.01
        self.adhesion_coeff = self.surface_tension_coeff
        self.time = 0.0


    def initialize(self):

        self.ns.broad_phase()
        self.ns.narrow_phase(self.ps.x, self.ps.particle_neighbors_num, self.ps.particle_neighbors)
        self.ns.narrow_phase_surface(self.ps.x_0_s, self.ps.x, self.ps.surface_neighbor_num, self.ps.surface_neighbor_idx)
        self.elasticity.initialize()


    @ti.kernel
    def apply_gravity(self, dt: float):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic[p_i]:
                self.ps.acceleration[p_i].fill(0.0)
                continue

            acc = ti.Vector(self.g)
            self.ps.acceleration[p_i] = acc
            self.ps.v[p_i] += acc * dt


    @ti.kernel 
    def advect_velocity(self, dt: float):
        # Symplectic Euler
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] = self.ps.v[p_i] + dt * self.ps.acceleration[p_i]
            else:
                self.ps.v[p_i] = ti.math.vec3(0.0)


    @ti.kernel
    def advect_position(self, dt: float):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                # self.ps.v[p_i] += self.dt * self.ps.acceleration[p_i]
                self.ps.x[p_i] += dt * self.ps.v[p_i]


    @ti.kernel
    def apply_damping(self, coeff: float):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] *= coeff


    def test(self):
        self.elasticity.test()


    def forward(self):

        # t0 = time.perf_counter()

        self.ns.broad_phase()
        self.ns.narrow_phase(self.ps.x, self.ps.particle_neighbors_num, self.ps.particle_neighbors)

        # elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # print("neighbour search: ", elapsed_ms)

        self.apply_gravity(self.dt)

        # self.surface_tension.solve(self.surface_tension_coeff, self.adhesion_coeff, self.dt)

        # self.viscosity.solve(self.viscosity_coeff, self.dt)

        self.elasticity.solve(self.dt)

        t0 = time.perf_counter()
        self.pressure.solve(self.dt)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print("pressure: ", elapsed_ms)

        self.advect_position(self.dt)

        self.ns.enforce_boundary_3D()

        # self.apply_damping(0.99)

        self.elasticity.update_surface_vertex()

