# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)
import math
import numpy as np
import taichi as ti


ti.init(arch=ti.gpu)

screen_res = (600, 800)
screen_to_world_ratio = 10.0

dim = 2
bg_color = 0x112F41
particle_color = 0x068587
boundary_color = 0xEBACA2
num_particles_x = 60
num_particles = num_particles_x * 100
max_num_particles_per_cell = 400
max_num_neighbors = 400
time_delta = 0.005
epsilon = 1e-5
eps = 1e-4
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 0.7
tol = 1e-3
mass = 1.0
rho0 = 10000.0
lambda_epsilon = 1e-1
max_jacobi_iter = 1000
corr_deltaQ_coeff = 0.3
corrK = 0.001

# Need ti.pow()
# corrN = 4.0
mu = 1e-5
neighbor_radius = h_ * 1.05

# Boundary particles
boundary_particle_spacing = h_ * 0.8  # Spacing between boundary particles
boundary_layers = 3  # Number of boundary layers


boundary = (
    screen_res[0] / screen_to_world_ratio,
    screen_res[1] / screen_to_world_ratio,
)
cell_size = 2.51 * h_
cell_recpr = 1.0 / cell_size

def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s


grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))

# Termination condition control
use_measure_error = True  # True: use measure_error(), False: use residual norm
num_boundary_x = int(boundary[0] / boundary_particle_spacing) + 1
num_boundary_y = int(boundary[1] / boundary_particle_spacing) + 1

# Compute boundary positions in numpy
def generate_box_particles(start_corner, end_corner, spacing):
    """Generate particles in a solid box between start_corner and end_corner"""
    particles = []
    x = start_corner[0]
    while x <= end_corner[0] + 1e-6:
        y = start_corner[1]
        while y <= end_corner[1] + 1e-6:
            particles.append([x, y])
            y += spacing
        x += spacing
    return particles

def generate_boundary_box(outer_start, outer_end, num_layers, spacing):
    """Generate boundary particles by creating outer box and removing inner box"""
    # Step 1: Create solid outer box
    outer_particles = generate_box_particles(outer_start, outer_end, spacing)
    
    # Step 2: Calculate inner box corners
    thickness = num_layers * spacing
    inner_start = [outer_start[0] + thickness, outer_start[1] + thickness]
    inner_end = [outer_end[0] - thickness, outer_end[1] - thickness]
    
    # Step 3: Remove inner box particles (if inner box is valid)
    boundary_particles = []
    if inner_start[0] < inner_end[0] and inner_start[1] < inner_end[1]:
        for pos in outer_particles:
            x, y = pos[0], pos[1]
            # Keep particle if it's outside the inner box
            if (x < inner_start[0] or x > inner_end[0] or 
                y < inner_start[1] or y > inner_end[1]):
                boundary_particles.append(pos)
    else:
        # If inner box is invalid, keep all particles (solid boundary)
        boundary_particles = outer_particles
    
    return boundary_particles

def compute_boundary_positions():
    """Compute boundary positions using simulation domain boundaries"""
    # Use simulation domain start/end positions
    domain_start = [0.0, 0.0]  # Bottom-left corner of simulation domain
    domain_end = [boundary[0], boundary[1]]  # Top-right corner of simulation domain
    
    # Add small offset to ensure boundary particles are within domain
    particle_radius_offset = 1.0
    outer_start = [domain_start[0] + particle_radius_offset, 
                   domain_start[1] + particle_radius_offset]
    outer_end = [domain_end[0] - particle_radius_offset, 
                 domain_end[1] - particle_radius_offset]
    
    positions = generate_boundary_box(outer_start, outer_end, boundary_layers, boundary_particle_spacing)
    return np.array(positions)

boundary_positions_np = compute_boundary_positions()
num_boundary_particles = len(boundary_positions_np)

# Debug: Check corner positions
def debug_boundary_corners():
    print("Boundary positions (first 20):")
    for i, pos in enumerate(boundary_positions_np[:20]):
        print(f"  {i}: [{pos[0]:.2f}, {pos[1]:.2f}]")
    print(f"Total boundary particles: {num_boundary_particles}")
    
    # Check if corners exist
    corners = [
        [0.0, 0.0],  # bottom-left
        [boundary[0], 0.0],  # bottom-right
        [0.0, boundary[1]],  # top-left
        [boundary[0], boundary[1]]  # top-right
    ]
    
    print("Corner check:")
    for i, corner in enumerate(corners):
        found = False
        for pos in boundary_positions_np:
            if abs(pos[0] - corner[0]) < 0.1 and abs(pos[1] - corner[1]) < 0.1:
                found = True
                break
        print(f"  Corner {corner}: {'Found' if found else 'MISSING'}")

debug_boundary_corners()
# 2D kernel normalization constants
poly6_factor = 315.0 / (64.0 * math.pi)  # 2D Poly6: 315/(64π)
spiky_grad_factor = -45.0 / math.pi       # 2D Spiky gradient: -45/π
# 2D cubic (Wendland) kernel constants
cubic_factor = 15.0 / (7.0 * math.pi)    # 2D Cubic: 15/(7π)
cubic_grad_factor = -45.0 / (7.0 * math.pi)  # 2D Cubic gradient: -45/(7π)
old_positions = ti.Vector.field(dim, float)
Aii = ti.field(float)
Dii = ti.field(float)
Ax = ti.field(float)
tmp = ti.Vector.field(dim, float)
rho = ti.field(float)
res = ti.field(float)
src = ti.field(float)
num = ti.field(float)
positions = ti.Vector.field(dim, float)
positions_normalized = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
velocities_tmp = ti.Vector.field(dim, float)
boundary_positions = ti.Vector.field(dim, float)
boundary_positions_normalized = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
particle_wij = ti.Vector.field(dim, float)
p = ti.field(float)
m = ti.field(float)
barrier_grad = ti.field(float)
barrier_hess = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# line-search fields
dp = ti.field(float)
omega_ls = ti.field(float, shape=())
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)
ti.root.dense(ti.i, num_particles).place(old_positions, positions, positions_normalized, velocities, velocities_tmp)
ti.root.dense(ti.i, num_boundary_particles).place(boundary_positions, boundary_positions_normalized)
ti.root.dense(ti.i, num_particles).place(Aii, Dii, src, Ax, rho, res, tmp, num)
ti.root.dense(ti.i, num_particles).place(dp)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors, particle_wij)
ti.root.dense(ti.i, num_particles).place(p, m, barrier_grad, barrier_hess, position_deltas)
ti.root.place(board_states)


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 <= s and s < h:
        # 2D Poly6: (315/(64πh⁹)) * (h² - r²)³
        h_sqr = h * h
        h_9 = h_sqr * h_sqr * h_sqr * h_sqr * h
        result = poly6_factor * (h_sqr - s * s) ** 3 / h_9
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        # 2D Spiky gradient: -(45/(πh⁶)) * (h - r)² * r̂
        h_6 = h * h * h * h * h * h
        g_factor = spiky_grad_factor * (h - r_len) * (h - r_len) / h_6
        result = r * g_factor / r_len
    return result


@ti.func
def cubic_value(s, h):
    result = 0.0
    if 0 <= s and s < h:
        # 2D Cubic (Wendland): W(r,h) = (15/(7πh²)) * (1 - r/h)³
        q = s / h
        h_2 = h * h
        result = cubic_factor * (1.0 - q) ** 3 / h_2
    return result


@ti.func
def cubic_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        # 2D Cubic gradient: ∇W(r,h) = -(45/(7πh³)) * (1 - r/h)² * r̂
        q = r_len / h
        h_3 = h * h * h
        g_factor = cubic_grad_factor * (1.0 - q) * (1.0 - q) / h_3
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = cubic_value(pos_ji.norm(), h_) / cubic_value(corr_deltaQ_coeff * h_, h_)
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
def move_board(dt: float):
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * dt
    board_states[None] = b


@ti.kernel
def neighbor_search(x: ti.template()):
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1
    # update grid
    for p_i in x:
        cell = get_cell(x[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in x:
        pos_i = x[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (pos_i - x[p_j]).norm() < neighbor_radius:

                        if p_j >= 0:
                            particle_neighbors[p_i, nb_i] = p_j
                            nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.kernel
def prologue(dt: float):
    # save old positions
    # for i in positions:
    #     old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8])
        # pos, vel = positions[i]
        velocities[i] += g * dt
        # pos += vel * dt
        # positions[i] = confine_position_to_boundary(pos)
    # clear neighbor lookup table


@ti.kernel
def update_lambdas(omega: float):
    for p_i in positions:
        p[p_i] = p[p_i] + omega * res[p_i] / (Aii[p_i] + lambda_epsilon) 
        # if pi <= 0.0:
        #     pi = eps

        # res[p_i] = src[p_i] - Ax[p_i]
        # # p[p_i] += omega * (res[p_i] / (Aii[p_i] + lambda_epsilon))
        # numer = res[p_i] + mu / pi
        # denom = Aii[p_i] + mu / (pi * pi) + lambda_epsilon
        # dp[p_i] = numer / denom
        # num[p_i] = numer
        # print("dp: ", dp[p_i])
        # print("boundary hessian: ", mu / (pi * pi))

@ti.kernel
def advect_positions(dt: float):
    for i in positions:
        positions[i] += dt * velocities[i]

@ti.kernel
def substep(dt: float):
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = p[p_i]
        f_i = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = p[p_j]
            pos_ji = pos_i - positions[p_j]
            # scorr_ij = compute_scorr(pos_ji)
            f_i -= m[p_j] * (lambda_i + lambda_j) * cubic_gradient(pos_ji, h_)
        # pos_delta_i /= rho0
        velocities[p_i] += dt * f_i / m[p_i]
    # apply position deltas
    for i in positions:
        positions[i] += dt * velocities[i]

@ti.kernel
def epilogue(dt: float):
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / dt
    # no vorticity/xsph because we cannot do cross product in 2D...

@ti.kernel
def precompute(dt: float):
        
    # print(m[0] * cubic_value(0.0, h_))
    for p_i in positions:
        pos_i = positions[p_i]
        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density = m[p_i] * cubic_value(0.0, h_)
        # print("")
        div = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            # if p_j < 0:
            #     break
            pos_ji = pos_i - positions[p_j]
            grad_j = cubic_gradient(pos_ji, h_)
            particle_wij[p_i, j] = grad_j
            div += m[p_j] * grad_j.dot(velocities[p_i] - velocities[p_j])
            grad_i += m[p_j] * grad_j
            sum_gradient_sqr += (m[p_j] * grad_j).dot(m[p_j] * grad_j) / m[p_j]
            density += m[p_j] * cubic_value(pos_ji.norm(), h_)
        sum_gradient_sqr += grad_i.dot(grad_i) / m[p_i]
        rho[p_i] = density
        Aii[p_i] = m[p_i] * sum_gradient_sqr + 1e-3
        src[p_i] = (density + dt * div - rho0) / (dt ** 2)


@ti.kernel
def compute_Ax_step_1(res: ti.template(), x: ti.template()):

    for p_i in positions:
        res[p_i] = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            grad_ij = particle_wij[p_i, j] 
            # if p_j < 0:
            #     break
            res[p_i] += m[p_i] * m[p_j] * (x[p_i] + x[p_j]) * grad_ij
            # Ax[p_i] += lambdas[p_j] * cubic_gradient(positions[p_i] - positions[p_j], h_)
    


@ti.kernel
def compute_Ax_step_2(res: ti.template(), x: ti.template()):
    for p_i in positions:
        res[p_i] = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            grad_ij = particle_wij[p_i, j] 
            # if p_j < 0:
            #     break
            res[p_i] += m[p_j] * (x[p_i] - x[p_j]).dot(grad_ij)
            # Ax[p_i] += lambdas[p_j] * cubic_gradient(positions[p_i] - positions[p_j], h_)


@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> float:

    ret = 0.0
    for i in a:
        ret += ti.math.dot(a[i], b[i])

    return ret

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


@ti.kernel
def jacobi_precondition(output: ti.template(), input: ti.template(), D_ii: ti.template()):
    """Apply Jacobi preconditioning: output[p_i] = input[p_i] / D_ii[p_i]"""
    for p_i in range(num_particles):
        diagonal_val = D_ii[p_i]
        
        # Ensure diagonal element is not zero
        if abs(diagonal_val) < eps:
            diagonal_val = eps  # Use small epsilon to avoid division by zero
            
        output[p_i] = input[p_i] / diagonal_val

@ti.kernel
def compute_barrier_grad_hess(p: ti.template(), p0: float):
    """Compute barrier gradient and Hessian for b(p) = -log(p/p0) * (p - p0)^2"""
    
    for p_i in range(num_particles):
        p_val = p[p_i]
        
        # Ensure p > 0 to avoid numerical issues
        if p_val <= 0.0:
            p_val = eps
            
        # Barrier function: b(p) = -log(p/p0) * (p - p0)^2
        # Let u = p - p0, v = log(p/p0)
        u = p_val - p0  # (p - p0)
        v = ti.log(p_val / p0)  # log(p/p0)
        
        # Gradient: ∇b(p) = -(1/p) * (p - p0)^2 - log(p/p0) * 2(p - p0)
        #                = -(u^2/p + 2*v*u)
        barrier_grad[p_i] = -(u * u / p_val + 2.0 * v * u)
        
        # Hessian: ∇²b(p) = d/dp[-(u^2/p + 2*v*u)]
        #                 = d/dp[-(p-p0)^2/p - 2*log(p/p0)*(p-p0)]
        #                 = (p-p0)^2/p^2 - 2*(p-p0)/p - 2*log(p/p0) - 2*(p-p0)/p
        #                 = (p-p0)^2/p^2 - 4*(p-p0)/p - 2*log(p/p0)
        barrier_hess[p_i] = u * u / (p_val * p_val) - 4.0 * u / p_val - 2.0 * v

@ti.kernel
def line_search_alpha(p: ti.template(), dp: ti.template()) -> float:
    """Line search for alpha such that p + alpha * dp >= eps for all particles"""
    min_alpha = 1.0
    
    for p_i in range(num_particles):
        if dp[p_i] < 0.0:
            # Calculate maximum alpha that keeps p_i + alpha * dp_i >= eps
            candidate_alpha = (eps - p[p_i]) / dp[p_i]
            
            # Only consider positive candidates (since dp_i < 0, candidate should be positive)
            if candidate_alpha > 0.0:
                ti.atomic_min(min_alpha, candidate_alpha)
    
    return min_alpha

@ti.kernel
def measure_error(v: ti.template(), dt: float) -> float:

    avg_error = 0.0
    for p_i in positions:
        pos_i = positions[p_i]
        div = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = cubic_gradient(pos_ji, h_)
            div += m[p_j] * grad_j.dot(v[p_i] - v[p_j])

        avg_error += ti.max(rho[p_i] + dt * div - rho0, 0.0) / rho0

    avg_error /= num_particles
    return avg_error


def run_pbf(dt):


    old_positions.copy_from(positions)
    prologue(dt)
    neighbor_search(positions)
    precompute(dt)
    
    
    p.fill(0.0)
    num_iter = 0

    for _ in range(max_jacobi_iter):

    
        compute_Ax_step_1(tmp, p)

        jacobi_precondition(tmp, tmp, m)

        compute_Ax_step_2(Ax, tmp)

        add(Ax, Ax, 1e-3, p)
        # compute src + Ax
        add(res, src, -1.0, Ax)
        
        # Choose termination condition based on GUI setting
        if use_measure_error:
            # Option 1: Use measure_error()
            add(velocities_tmp, velocities, -dt, tmp)
            err = measure_error(velocities_tmp, dt)
            if err < tol:
                print(f"Converged (measure_error) after {num_iter} iterations with error {err:.6f}")
                break
        else:
            # Option 2: Use residual norm
            err = ti.sqrt(dot2(res, res) / num_particles)
            if err < tol:
                print(f"Converged (residual norm) after {num_iter} iterations with error {err:.6f}")
                break

        Dii.copy_from(Aii)
        jacobi_precondition(dp, res, Dii)
        
        # Line search for step size
        # alpha = line_search_alpha(p, dp)
        add(p, p, 0.5, dp)
        
        project(p)
        num_iter += 1

    print("Jacobi iter: ", num_iter)
    # substep(dt)
    velocities.copy_from(velocities_tmp)
    advect_positions(dt)
    epilogue(dt)


@ti.kernel
def normalize_positions():
    for i in positions:
        positions_normalized[i][0] = positions[i][0] / boundary[0]
        positions_normalized[i][1] = positions[i][1] / boundary[1]
    
    for i in boundary_positions:
        boundary_positions_normalized[i][0] = boundary_positions[i][0] / boundary[0]
        boundary_positions_normalized[i][1] = boundary_positions[i][1] / boundary[1]

def render(window):
    canvas = window.get_canvas()
    canvas.set_background_color(((bg_color >> 16 & 0xff) / 255.0, 
                                (bg_color >> 8 & 0xff) / 255.0, 
                                (bg_color & 0xff) / 255.0))
    
    normalize_positions()
    
    # Render fluid particles
    canvas.circles(positions_normalized, radius=0.002, 
                  color=((particle_color >> 16 & 0xff) / 255.0,
                         (particle_color >> 8 & 0xff) / 255.0,
                         (particle_color & 0xff) / 255.0))
    
    # Render boundary particles in gray
    canvas.circles(boundary_positions_normalized, radius=0.004, 
                  color=(0.5, 0.5, 0.5))


@ti.kernel
def init_particles():

    alpha = (0.3 / cubic_value(0.0, h_)) 
    for i in range(num_particles):
        delta = h_ * 0.4
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5, boundary[1] * 0.02])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = 0.0
        m[i] = rho0 * alpha
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])



def init_boundary_particles():
    boundary_positions.from_numpy(boundary_positions_np)

def regenerate_boundary():
    global boundary_positions_np, num_boundary_particles
    boundary_positions_np = compute_boundary_positions()
    num_boundary_particles = len(boundary_positions_np)
    # Reallocate Taichi field if size changed
    boundary_positions.from_numpy(boundary_positions_np)
    print(f"Regenerated boundary: {num_boundary_particles} particles, spacing={boundary_particle_spacing:.2f}, layers={boundary_layers}")


def print_stats():
    print("PBF stats:")
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")


def reset():
    init_particles()
    p.fill(0.0)


def stop():
    return False


def show_options(gui, frame_cnt):
    global boundary_particle_spacing, boundary_layers, use_measure_error, time_delta, tol, max_jacobi_iter
    with gui.sub_window("Settings", 0., 0., 0.4, 0.4):
        gui.text(f"Current frame: {frame_cnt}")
        gui.text("")  # Spacer
        
        gui.text("Time Step:")
        time_delta = gui.slider_float("dt", time_delta, 0.001, 0.1)
        
        gui.text("")  # Spacer
        gui.text("Solver:")
        max_jacobi_iter = gui.slider_int("Max iterations", max_jacobi_iter, 1, 2000)
        
        gui.text("")  # Spacer
        gui.text("Convergence:")
        tol = gui.slider_float("Tolerance", tol, 1e-6, 1e-1)
        
        gui.text("")  # Spacer
        gui.text("Termination Condition:")
        use_measure_error = gui.checkbox("Use measure_error()", use_measure_error)
        if not use_measure_error:
            gui.text("Using residual norm")
        
        # gui.text("")  # Spacer
        # gui.text("Boundary Settings:")
        
        # new_spacing = gui.slider_float("Particle spacing", boundary_particle_spacing, 0.1, 2.0)
        # new_layers = gui.slider_int("Boundary layers", boundary_layers, 1, 10)
        
        # # Check if boundary settings changed
        # if abs(new_spacing - boundary_particle_spacing) > 1e-6 or new_layers != boundary_layers:
        #     boundary_particle_spacing = new_spacing
        #     boundary_layers = new_layers
        #     if gui.button("Regenerate boundary"):
        #         regenerate_boundary()

def main():
    init_particles()
    init_boundary_particles()
    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")
    print(f"num_boundary_particles={num_boundary_particles}")
    print(f"boundary_layers={boundary_layers}")
    
    window = ti.ui.Window('PBF2D', screen_res, show_window=True, vsync=False)
    gui = window.get_gui()
    
    runSim = True
    frame_cnt = 0
    
    while window.running:
        # Handle keyboard events
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                runSim = not runSim
                print(runSim)
            elif window.event.key == 'r':
                reset()
                runSim = False
                frame_cnt = 0
            elif window.event.key == ti.ui.ESCAPE:
                break
        
        show_options(gui, frame_cnt)
        
        if runSim:
            # move_board(time_delta)
            run_pbf(time_delta)
            frame_cnt += 1
            
        # if frame_cnt % 20 == 1:
        #     print_stats()
        
        render(window)
        window.show()

        
if __name__ == "__main__":
    main()