import taichi as ti
import pressure
from math_utils import *

class DFSPH(pressure):
    def __init__(self, particle_system):
        super().__init__(particle_system)

    def solve(self, dt):

        self.ps.x_old.copy_from(self.ps.x)
        add(self.ps.y, self.ps.x, dt, self.ps.v)

        self.ps.x.copy_from(self.ps.y)
        self.compute_density()
        self.compute_constraint()
        self.compute_Aii()

        add(self.s, self.ps.y, -1.0, self.ps.x)
        Jd = self.Jx
        d = self.dx
        d.copy_from(self.s)
        c = self.c
        g = self.tmp
        p = self.a
        opt_iter = 0
        for _ in range(self.max_iteration_opt):

            self.compute_J_x(Jd, d)
            add(self.t, Jd, 1.0, c)

            self.compute_f(self.f, self.t, self.eps)
            self.compute_f_derivative(self.dfdt, self.t, self.eps)
            coef_wise_mul(self.f, self.f, self.k)
            self.compute_J_tr_x(g, self.f)

            p.fill(0.0)

            self.PCG(x=p, b=g)
            add(d, d, -self.omega, p)

            if self.density_error:
                err = self.compute_avg_density_error(self.f)
            else:
                err = self.dot(p, p)
            err_log = self.dot(p, p)

            self.stats_opt_error.append(float(err_log))
            if self.print_opt_error:
                print(f"opt error: {err_log}")

            # Count this iteration before break check so that opt_iter matches logged errors
            opt_iter += 1
            if ((err < pow(10, -self.tol_opt)) and (opt_iter > 2)) or (opt_iter >= self.max_iteration_opt):
                break

        # x_n+1
        add(self.ps.x, self.ps.x, 1.0, self.dx)
        # v_n+1_tmp
        self.update_velocities(dt)
        self.ps.x.copy_from(self.ps.x_old)