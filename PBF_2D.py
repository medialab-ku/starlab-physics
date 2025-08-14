# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)
import math
import numpy as np
import taichi as ti


ti.init(arch=ti.gpu)

screen_res = (800, 400)
screen_to_world_ratio = 10.0
boundary = (
    screen_res[0] / screen_to_world_ratio,
    screen_res[1] / screen_to_world_ratio,
)
cell_size = 2.51
cell_recpr = 1.0 / cell_size

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))
dim = 2
bg_color = 0x112F41
particle_color = 0x068587
boundary_color = 0xEBACA2
num_particles_x = 60
num_particles = num_particles_x * 20
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 0.02
epsilon = 1e-5
eps = 1e-4
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 1e-3
max_jacobi_iter = 10
corr_deltaQ_coeff = 0.3
corrK = 0.001

# Need ti.pow()
# corrN = 4.0
mu = 1e-5
neighbor_radius = h_ * 1.05
poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi
old_positions = ti.Vector.field(dim, float)
Aii = ti.field(float)
Ax = ti.field(float)
tmp = ti.Vector.field(dim, float)
res = ti.field(float)
src = ti.field(float)
num = ti.field(float)
positions = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
p = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# line-search fields
dp = ti.field(float)
omega_ls = ti.field(float, shape=())
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)
ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)
ti.root.dense(ti.i, num_particles).place(Aii, src, Ax, res, tmp, num)
ti.root.dense(ti.i, num_particles).place(dp)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(p, position_deltas)
ti.root.place(board_states)


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_, h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x


@ti.func
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1]


@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]]) - particle_radius_in_world
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)
    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1
    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i


@ti.kernel
def update_lambdas():
    for p_i in positions:
        pi = p[p_i]
        if pi <= 0.0:
            pi = eps

        res[p_i] = src[p_i] - Ax[p_i]
        # p[p_i] += omega * (res[p_i] / (Aii[p_i] + lambda_epsilon))
        numer = res[p_i] + mu / pi
        denom = Aii[p_i] + mu / (pi * pi) + lambda_epsilon
        dp[p_i] = numer / denom
        num[p_i] = numer
        # print("dp: ", dp[p_i])
        # print("boundary hessian: ", mu / (pi * pi))



@ti.kernel
def substep():
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = p[p_i]
        pos_delta_i = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = p[p_j]
            pos_ji = pos_i - positions[p_j]
            # scorr_ij = compute_scorr(pos_ji)
            pos_delta_i -= (lambda_i + lambda_j) * spiky_gradient(pos_ji, h_)
        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]


@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    # no vorticity/xsph because we cannot do cross product in 2D...


@ti.kernel
def precompute():

     for p_i in positions:
        pos_i = positions[p_i]
        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density = poly6_value(0.0, h_)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            density += poly6_value(pos_ji.norm(), h_)
        sum_gradient_sqr += grad_i.dot(grad_i)
        Aii[p_i] = sum_gradient_sqr
        src[p_i] = (density - 1.0)


@ti.kernel
def compute_Ax():

    for p_i in positions:
        tmp[p_i] = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            tmp[p_i] += (p[p_i] + p[p_j]) * spiky_gradient(positions[p_i] - positions[p_j], h_)
            # Ax[p_i] += lambdas[p_j] * spiky_gradient(positions[p_i] - positions[p_j], h_)
    
    for p_i in positions:
        Ax[p_i] = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            Ax[p_i] += (tmp[p_i] - tmp[p_j]).dot(spiky_gradient(positions[p_i] - positions[p_j], h_))
            # Ax[p_i] += lambdas[p_j] * spiky_gradient(positions[p_i] - positions[p_j], h_)


@ti.kernel
def dot2(a: ti.template(), b: ti.template()) -> float:

    ret = 0.0
    for i in a:
        ret += (a[i] * b[i])

    return ret


@ti.kernel
def project(p: ti.template()):
    for p_i in ti.grouped(p):
        p[p_i] = ti.max(p[p_i], 0.0)


@ti.kernel
def add(ret: ti.template(), v0: ti.template(), scale: float, v1: ti.template()):
    for i in ret:
        ret[i] = v0[i] + scale * v1[i]


def run_pbf():
    prologue()

    precompute()
    p.fill(0.0)
    
    num_iter = 0
    for _ in range(max_jacobi_iter):

        # compute Ax 
        compute_Ax()

        # compute src + Ax
        # add(res2, src, 1.0, Ax)


        update_lambdas()
        # line-search for omega to keep p + omega*dp >= eps
        omega_ls[None] = 1.0
        for i in range(num_particles):
            # if p_i + dp_i < eps -> omega_i = (eps - p_i) / dp_i (only if dp_i < 0)
            if dp[i] < 0:
                cand = (eps - p[i]) / dp[i]
                # cand is positive when p[i] + cand*dp[i] == eps
                if cand < omega_ls[None]:
                    omega_ls[None] = cand
        # clamp omega in (0, 1]
        if omega_ls[None] <= 0 or omega_ls[None] > 1.0:
            omega_ls[None] = 1.0
        # apply update with omega_ls
        for i in range(num_particles):
            p[i] += omega_ls[None] * dp[i]
        project(p)


        err = ti.sqrt(dot2(dp, dp) / num_particles)
        # err = ti.sqrt(dot2(res, res) / num_particles)
        print(f"err: {err}")

        if err < 1e-3:
            break
            
        num_iter += 1

    print("Jacobi iter: ", num_iter)
    substep()
    epilogue()


def render(gui):
    gui.clear(bg_color)
    pos_np = positions.to_numpy()
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    gui.rect(
        (0, 0),
        (board_states[None][0] / boundary[0], 1),
        radius=1.5,
        color=boundary_color,
    )
    gui.show()


@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5, boundary[1] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = (ti.random() - 0.5) * 4
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])


def print_stats():
    print("PBF stats:")
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")


def main():
    init_particles()
    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")
    gui = ti.GUI("PBF2D", screen_res)
    while gui.running and not gui.get_event(gui.ESCAPE):
        move_board()
        run_pbf()
        if gui.frame % 20 == 1:
            print_stats()
        render(gui)

        
if __name__ == "__main__":
    main()