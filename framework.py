import taichi as ti

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


        self.YM = 1e1 # Young Modulus
        self.PR = 0.0 # Poisson Ratio


    def initialize(self):

        self.ns.broad_phase()
        self.ns.narrow_phase(self.ps.x)
        self.elasticity.initialize()
        # self.compute_static_boundary_volume()
        # self.compute_moving_boundary_volume()
        # self.ps.initialize_boundary_neighbors()

        # if self.ps.num_rigid_bodies > 0:
        #     self.ps.initialize_rigid_mass()

        # if hasattr(self.ps, "emitter_system") and self.ps.emitter_system:
        #     self.ps.emitter_system.reset()


    @ti.kernel
    def apply_gravity(self, dt: float):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid(p_i):
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


    def forward(self):

        # self.ns.broad_phase()
        # self.ns.narrow_phase(self.ps.x)

        self.apply_gravity(self.dt)

        # self.surface_tension.solve(self.surface_tension_coeff, self.adhesion_coeff, self.dt)

        # self.viscosity.solve(self.viscosity_coeff, self.dt)

        self.elasticity.solve(self.YM, self.PR, self.dt)

        # self.pressure.solve(self.dt)

        self.advect_position(self.dt)

        self.ns.enforce_boundary_3D()

