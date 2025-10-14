# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)
import math
import numpy as np
import taichi as ti


# print("test")

ti.init(arch=ti.gpu)

screen_res = (500, 1000)
screen_to_world_ratio = 400.0

solver_method = 0
dim = 2
bg_color = 0x112F41
particle_color = 0x068587
boundary_color = 0xEBACA2
num_particles_x = 60
num_particles = num_particles_x * 100
max_num_particles_per_cell = 400
max_num_neighbors = 800
time_delta = 0.001
epsilon = 1e-5
eps = 1e-2
particle_radius = 3.0


# PBF params
h_ = 0.02
particle_radius_in_world = h_ / 10
tol = 1e-2
mass = 1.0
# rho0 = 1000.0  # Now replaced with per-particle rho0 field
lambda_epsilon = 1e-1
max_jacobi_iter = 1000
max_pcg_iter = 1000
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
    particle_radius_offset = 0.1
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
Hii = ti.Matrix.field(dim, dim, float)
Dii = ti.field(float)
Ax = ti.field(float)
tmp = ti.Vector.field(dim, float)
rho = ti.field(float)
rho0 = ti.field(float)  # Per-particle rest density
res = ti.field(float)
src = ti.field(float)
num = ti.field(float)
positions = ti.Vector.field(dim, float)
positions_adv = ti.Vector.field(dim, float)
positions_normalized = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
velocities_adv = ti.Vector.field(dim, float)
velocities_tmp = ti.Vector.field(dim, float)
dx = ti.Vector.field(dim, float)
Ap_vec = ti.Vector.field(dim, float)
r_vec = ti.Vector.field(dim, float)
p_vec = ti.Vector.field(dim, float)
z_vec = ti.Vector.field(dim, float)
rhs_vec = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
particle_wij = ti.Vector.field(dim, float)
p = ti.field(float)
m = ti.field(float)
r = ti.field(float)
z = ti.field(float)
b = ti.field(float)
barrier_grad = ti.field(float)
barrier_hess = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# line-search fields
dp = ti.field(float)
omega_ls = ti.field(float, shape=())
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)
particle_is_dynamic = ti.field(dtype=ti.i32)

#TODO: allocate taichi data structure for boundary particle: neighbor search, mass

boundary_positions = ti.Vector.field(dim, float)
boundary_positions_normalized = ti.Vector.field(dim, float)

boundary_num_neighbors = ti.field(int)
boundary_neighbors = ti.field(int)

m_b = ti.field(float)
ti.root.dense(ti.i, num_boundary_particles).place(boundary_positions, boundary_positions_normalized)
ti.root.dense(ti.i, num_boundary_particles).place(m_b)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, positions_adv, positions_normalized, dx, velocities, velocities_adv, velocities_tmp, particle_is_dynamic)
ti.root.dense(ti.i, num_particles).place(Ap_vec, r_vec, p_vec, z_vec, rhs_vec)
ti.root.dense(ti.i, num_particles).place(Aii, Dii, Hii, src, r, z, Ax, rho, rho0, res, tmp, num, b)
ti.root.dense(ti.i, num_particles).place(dp)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(boundary_num_neighbors)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(boundary_neighbors, particle_neighbors, particle_wij)
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


#TODO: separate neighbor search into two steps
@ti.kernel
def neighbor_search(x: ti.template(), x_b: ti.template()):
    
    # step 1: clear grid
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    # for I in ti.grouped(particle_neighbors):
    #     particle_neighbors[I] = -1
    # update grid

    # step 2: update particles in cell
    for p_i in x:
        cell = get_cell(x[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i

    for p_i in x_b:
        cell = get_cell(x_b[p_i])
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        # Add offset to distinguish boundary particles
        grid2particles[cell, offs] = p_i + num_particles


    for i in range(num_particles):
        particle_num_neighbors[i] = 0
        boundary_num_neighbors[i] = 0

    # step 3: find particle neighbors & boundary neighbors
    for p_i in x:
        pos_i = x[p_i]
        cell = get_cell(pos_i)
        
        nb_i = 0
        nbb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j_global = grid2particles[cell_to_check, j]
                    is_fluid = p_j_global < num_particles

                    if is_fluid:
                        p_j_local = p_j_global
                        if p_i != p_j_local and (pos_i - x[p_j_local]).norm() < neighbor_radius:
                            if nb_i < max_num_neighbors:
                                particle_neighbors[p_i, nb_i] = p_j_local
                                nb_i += 1
                    else:
                        p_k_local = p_j_global - num_particles
                        if (pos_i - x_b[p_k_local]).norm() < neighbor_radius:
                            if nbb_i < max_num_neighbors:
                                boundary_neighbors[p_i, nbb_i] = p_k_local
                                nbb_i += 1
            
            particle_num_neighbors[p_i] = nb_i
            boundary_num_neighbors[p_i] = nbb_i

@ti.kernel
def advect_velocity(dt: float):
    # save old positions
    # for i in positions:
    #     old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.8])
        # pos, vel = positions[i]
        velocities_adv[i] = velocities[i] + g * dt
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
def project_boundary(x: ti.template()):

    for i in positions:
        pos = x[i]
        x[i] = confine_position_to_boundary(pos)
@ti.kernel
def epilogue(dt: float):
    # confine to boundary
    # for i in positions:
    #     pos = positions[i]
    #     positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / dt
    # no vorticity/xsph because we cannot do cross product in 2D...

@ti.kernel
def precompute(x: ti.template(), x_b: ti.template()):
        
    # print(m[0] * cubic_value(0.0, h_))

    # for fluid particles
    for p_i in x:
        pos_i = x[p_i]
        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density = m[p_i] * cubic_value(0.0, h_)
        # print("")
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            # if p_j < 0:
            #     break
            pos_ji = pos_i - x[p_j]
            grad_j = cubic_gradient(pos_ji, h_)
            particle_wij[p_i, j] = grad_j
            grad_i += m[p_j] * grad_j
            sum_gradient_sqr += (m[p_j] * grad_j).dot(m[p_j] * grad_j) / m[p_j]
            density += m[p_j] * cubic_value(pos_ji.norm(), h_)

        #TODO: for boundary particles: density, Aii
        for k in range(boundary_num_neighbors[p_i]):
            p_k = boundary_neighbors[p_i, k]
            pos_ki = pos_i - x_b[p_k]
            grad_k = cubic_gradient(pos_ki, h_)
            particle_wij[p_i, particle_num_neighbors[p_i] + k] = grad_k
            grad_i += m_b[p_k] * grad_k
            density += m_b[p_k] * cubic_value(pos_ki.norm(), h_)

        sum_gradient_sqr += grad_i.dot(grad_i) / m[p_i]
        rho[p_i] = density
        Aii[p_i] = sum_gradient_sqr + 1e-3


@ti.kernel
def precompute_vanilla(x: ti.template()):

    # print(m[0] * cubic_value(0.0, h_))
    for p_i in x:
        pos_i = x[p_i]
        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density = m[p_i] * cubic_value(0.0, h_)
        # print("")
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            # if p_j < 0:
            #     break
            pos_ji = pos_i - x[p_j]
            grad_j = cubic_gradient(pos_ji, h_)
            particle_wij[p_i, j] = grad_j
            grad_i += m[p_j] * grad_j
            sum_gradient_sqr += (m[p_j] * grad_j).dot(m[p_j] * grad_j) / m[p_j]
            density += m[p_j] * cubic_value(pos_ji.norm(), h_)

        rho[p_i] = density
        sum_gradient_sqr += grad_i.dot(grad_i) / m[p_i]
        Aii[p_i] = (m[p_i] / rho[p_i] ** 2) * sum_gradient_sqr + 1e-3
        # src[p_i] = (density + dt * div - rho0) / (dt ** 2)

@ti.kernel
def compute_src(src: ti.template(), v: ti.template(), dt: float):

     for p_i in positions:
        div = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            # if p_j < 0:
            #     brea
            grad_j = particle_wij[p_i, j]
            div += m[p_j] * grad_j.dot(v[p_i] - v[p_j])
        src[p_i] = (rho[p_i] + dt * div - rho0[p_i]) / (dt ** 2)

@ti.kernel
def compute_J_tr_x(res: ti.template(), x: ti.template()):

    for p_i in positions:
        res[p_i] = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            grad_ij = particle_wij[p_i, j] 
            # if p_j < 0:
            #     break
            res[p_i] += (m[p_j] * x[p_i] + m[p_i] * x[p_j]) * grad_ij
            # Ax[p_i] += lambdas[p_j] * cubic_gradient(positions[p_i] - positions[p_j], h_)

        #TODO: for boundary particles: compute_J_tr_x   
        for k in range(boundary_num_neighbors[p_i]):
            p_k = boundary_neighbors[p_i, k]
            grad_ik = particle_wij[p_i, particle_num_neighbors[p_i] + k]
            res[p_i] += (m_b[p_k] * x[p_i]) * grad_ik
    

@ti.kernel
def compute_J_x(res: ti.template(), x: ti.template()):
    for p_i in positions:
        res[p_i] = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            grad_ij = particle_wij[p_i, j] 
            # if p_j < 0:
            #     break
            res[p_i] += m[p_j] * (x[p_i] - x[p_j]).dot(grad_ij)
            # Ax[p_i] += lambdas[p_j] * cubic_gradient(positions[p_i] - positions[p_j], h_)

        #TODO: for boundary particles: compute_J_x
        for k in range(boundary_num_neighbors[p_i]):
            p_k = boundary_neighbors[p_i, k]
            grad_ik = particle_wij[p_i, (particle_num_neighbors[p_i] + k)]
            res[p_i] += m_b[p_k] * x[p_i].dot(grad_ik)

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
def measure_error2(v: ti.template()) -> float:

    avg_error = 0.0
    for p_i in positions:
        avg_error += v[p_i] / rho0[p_i]

    avg_error /= num_particles
    return avg_error

@ti.kernel
def measure_error(v: ti.template()) -> float:

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

        avg_error += ti.max(rho[p_i] + div - rho0[p_i], 0.0) / rho0[p_i]

    avg_error /= num_particles
    return avg_error


def run_iisph(dt):


    old_positions.copy_from(positions)
    advect_velocity(dt)
    neighbor_search(positions, boundary_positions)
    precompute(positions, boundary_positions)
    velocities.copy_from(velocities_adv)
    
    compute_src(src, velocities, dt)
    p.fill(0.0)
    num_iter = 0

    for _ in range(max_jacobi_iter):
    
        compute_J_tr_x(tmp, p)
        jacobi_precondition(tmp, tmp, m)
        
        add(velocities, velocities_adv, -dt, tmp)

        dx.fill(0.0)
        add(dx, dx, dt, velocities)

        err = measure_error(dx)
        if err < tol:
            print(f"Converged (measure_error) after {num_iter} iterations with error {err:.6f}")
            break


        compute_J_x(Ax, tmp)
        # add(Ax, Ax, 1e-3, p)
        add(res, src, -1.0, Ax)
 
        jacobi_precondition(dp, res, Aii)
        add(p, p, 0.5, dp)
        
        project(p)
        num_iter += 1

    # print("Jacobi iter: ", num_iter)
    advect_positions(dt)
    project_boundary(positions)
    epilogue(dt)

@ti.kernel
def compute_M_inv_J_tr_D_x(res: ti.template(), x: ti.template()):

    """Compute M_inv * J^T * D * x"""
    for p_i in positions:
        res[p_i] = ti.math.vec2(0.0, 0.0)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            grad_ij = particle_wij[p_i, j]
            if p_j < 0:
                break
            res[p_i] += m[p_j] * (x[p_i] / (rho[p_i] ** 2) + x[p_j] / (rho[p_j] ** 2)) * grad_ij


def run_iisph_vanilla(dt):

    old_positions.copy_from(positions)
    advect_velocity(dt)
    neighbor_search(positions)
    precompute_vanilla(positions)
    velocities.copy_from(velocities_adv)

    compute_src(src, velocities, dt)
    p.fill(0.0)
    num_iter = 0

    for _ in range(max_jacobi_iter):

        compute_M_inv_J_tr_D_x(tmp, p)
        add(velocities, velocities_adv, -dt, tmp)

        dx.fill(0.0)
        add(dx, dx, dt, velocities)

        err = measure_error(dx)
        if err < tol:
            print(f"Converged (measure_error) after {num_iter} iterations with error {err:.6f}")
            break

        compute_J_x(Ax, tmp)
        add(res, src, -1.0, Ax)

        jacobi_precondition(dp, res, Aii)
        add(p, p, 0.5, dp)

        project(p)
        num_iter += 1

    # print("Jacobi iter: ", num_iter)
    advect_positions(dt)
    project_boundary(positions)
    epilogue(dt)


@ti.kernel
def compute_density_and_Aii(x: ti.template()):

    for p_i in x:
        pos_i = x[p_i]
        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        test = 0.0 
        density = m[p_i] * cubic_value(0.0, h_)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            pos_ji = pos_i - positions[p_j]
            grad_j = cubic_gradient(pos_ji, h_)
            particle_wij[p_i, j] = grad_j
            grad_i += m[p_j] * grad_j
            sum_gradient_sqr += (m[p_j] * grad_j).dot(m[p_j] * grad_j) / m[p_j]
            test += (m[p_j] * grad_j).dot(m[p_j] * grad_j)
            density += m[p_j] * cubic_value(pos_ji.norm(), h_)
        sum_gradient_sqr += grad_i.dot(grad_i) / m[p_i]
        test += grad_i.dot(grad_i) 
        rho[p_i] = density
        Aii[p_i] = sum_gradient_sqr  + 1e-3
        Dii[p_i] = test


@ti.kernel
def measure_error_pbf(src: ti.template()) -> float:

    avg_error = 0.0
    for p_i in positions:

        avg_error += src[p_i] / rho0[p_i]

    avg_error /= num_particles
    return avg_error


@ti.kernel
def compute_test(dtSq: float):

    #goal: (M + k * dt^2 * J^t J) * x = (M * y - k * dtSq * J^t c)
    """Compute test values for debugging"""
    k = 1e-9

    # compute c(x), activated when >=0  
    for p_i in positions:
        src[p_i] = ti.max(rho[p_i] - rho0[p_i], 0.0)

    # compute J^t c(x) and 2x2 block diagonal elements of J^t J
    for p_i in positions:
        tmp[p_i] = ti.math.vec2(0.0)
        ggT = ti.math.mat2(0.0)
        g_sum = ti.math.vec2(0.0)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            grad_j = particle_wij[p_i, j]
            tmp[p_i] += (m[p_j] * src[p_i] + m[p_i] * src[p_j]) * grad_j
            g_sum += m[p_j] * grad_j
            ggT += m[p_i] * m[p_i] * (grad_j.outer_product(grad_j))
    
        Hii[p_i] = ggT + g_sum.outer_product(g_sum)

    # A = M + k * dt^2 * J^t J
    id2 = ti.math.mat2([[1.0, 0.0], [0.0, 1.0]])
    for p_i in positions:
        b[p_i] = m[p_i] * positions_adv[p_i] + k * dtSq * tmp[p_i]
        Hii[p_i] = m[p_i] * id2 + k * dtSq * Hii[p_i]


    # apply_A_diag_only(x, Ap)          # Ap = A x
    # axpy(r, 1.0, b, -1.0, Ap)         # r = b - Ap

    # precond_diag(r, z)

    # copy_vec(p, z)

    # rz_old = dot2(r, z)
    # if rz_old < eps:
    #     return

    # for _ in range(max_pcg_iter):

    #     apply_A_diag_only(p, Ap)

    #     pAp = dot2(p, Ap)
    #     # 방어적 분모 클램프
    #     if abs(pAp) < 1e-20:
    #         break

    #     alpha = rz_old / pAp

    #     # x_{k+1} = x_k + alpha p_k
    #     add(x, x, +alpha, p)

    #     # r_{k+1} = r_k - alpha A p_k
    #     add(r, r, -alpha, Ap)

    #     rr = dot2(r, r)
    #     if rr < eps:
    #         break

    #     # z_{k+1} = M^{-1} r_{k+1}  (M^{-1}≈Aii^{-1})
    #     precond_diag(r, z)

    #     rz_new = dot2(r, z)
    #     beta = rz_new / (rz_old + 1e-20)

    #     # p_{k+1} = z_{k+1} + beta p_k
    #     add(p, z, beta, p)

    #     rz_old = rz_new

    # copy_vec(positions, x)



@ti.kernel
def finalize_A(out_vec: ti.template(), x_vec: ti.template(), Ap_vec: ti.template(), k_val: float, dt2: float):
    # out = M x + k dt^2 Ap
    for p_i in positions:
        out_vec[p_i] = m[p_i] * x_vec[p_i] + k_val * dt2 * Ap_vec[p_i]


@ti.kernel
def apply_Minv(out_vec: ti.template(), in_vec: ti.template()):
    # Jacobi preconditioner: use block-diagonal inverse Hii^{-1}
    for p_i in positions:
        out_vec[p_i] = Hii[p_i].inverse() @ in_vec[p_i]


@ti.kernel
def build_rhs(k_val: float, dt2: float):
    for p_i in positions:
        rhs_vec[p_i] = m[p_i] * (positions_adv[p_i]- positions[p_i]) + (-k_val * dt2) * tmp[p_i]

@ti.kernel
def build_block_diag_and_JtC(k: float, dtSq: float):
    # compute c(x), activated when >=0
    for p_i in positions:
        src[p_i] = ti.max(rho[p_i] - rho0[p_i], 0.0)

    # compute J^t c(x) and 2x2 block diagonal elements of J^t J
    for p_i in positions:
        tmp[p_i] = ti.math.vec2(0.0)
        ggT = ti.math.mat2(0.0)
        g_sum = ti.math.vec2(0.0)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            grad_j = particle_wij[p_i, j]
            tmp[p_i] += (m[p_j] * src[p_i] + m[p_i] * src[p_j]) * grad_j
            g_sum += m[p_j] * grad_j
            ggT += m[p_i] * m[p_i] * (grad_j.outer_product(grad_j))

        Hii[p_i] = ggT + g_sum.outer_product(g_sum)

    # A = M + k * dt^2 * J^t J
    # x = diag3x3 (A) ^-1 * (M * y - k * dtSq * J^t c)
    id2 = ti.math.mat2([[1.0, 0.0], [0.0, 1.0]])
    for p_i in positions:

        tmp[p_i] = m[p_i] * (positions_adv[p_i]-positions[p_i]) - k * dtSq * tmp[p_i]
        Hii[p_i] = m[p_i] * id2 + k * dtSq * Hii[p_i]

@ti.kernel
def compute_test(dtSq: float):


    #goal: (M + k * dt^2 * J^t J) * x = (M * y - k * dtSq * J^t c) 
    
    """Compute test values for debugging"""
    k = 1e-7

    # compute c(x), activated when >=0  
    for p_i in positions:
        src[p_i] = ti.max(rho[p_i] - rho0[p_i], 0.0)

    # compute J^t c(x) and 2x2 block diagonal elements of J^t J
    for p_i in positions:
        tmp[p_i] = ti.math.vec2(0.0)
        ggT = ti.math.mat2(0.0)
        g_sum = ti.math.vec2(0.0)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            grad_j = particle_wij[p_i, j]
            tmp[p_i] += (m[p_j] * src[p_i] + m[p_i] * src[p_j]) * grad_j
            g_sum += m[p_j] * grad_j
            ggT += m[p_i] * m[p_i] * (grad_j.outer_product(grad_j))
    
        Hii[p_i] = ggT + g_sum.outer_product(g_sum)

    # A = M + k * dt^2 * J^t J
    # x = diag3x3 (A) ^-1 * (M * y - k * dtSq * J^t c) 
    id2 = ti.math.mat2([[1.0, 0.0], [0.0, 1.0]])
    for p_i in positions:

        grad = m[p_i] * (positions[p_i] - positions_adv[p_i]) - k * dtSq * tmp[p_i]
        Hii[p_i] = m[p_i] * id2 + k * dtSq * Hii[p_i]
        positions[p_i] = positions[p_i] - Hii[p_i].inverse() @ grad

def run_pbf(dt):

    old_positions.copy_from(positions)
    advect_velocity(dt)
    velocities.copy_from(velocities_adv)
    add(positions_adv, positions, dt, velocities)

    positions.copy_from(positions_adv)
    project_boundary(positions)
    neighbor_search(positions)
    
    num_iter = 0
    dtSq = dt * dt
    for _ in range(max_jacobi_iter):

        compute_density_and_Aii(positions)

        # add(src, rho, -1.0, rho0)
        # velocities_tmp.fill(0.0)
        # compute_src(src, velocities_tmp, 1.0)
        
        # add(velocities_tmp, positions_adv, -1.0, positions)
        # compute_J_x(Ax, velocities_tmp)
        # add(src, src, -1.0, Ax)

        # project(src)
        # err = measure_error2(src)
        # if err < tol:
        #     print(f"Converged (measure_error) after {num_iter} iterations with error {err:.6f}")
        #     break


        # jacobi_precondition(p, src, Aii)
        # project(p)

        # compute_J_tr_x(tmp, p)
        # jacobi_precondition(tmp, tmp, m)

        # dx.fill(0.0) 
        # add(dx, dx, -1.0, tmp)


        # add(positions, positions, -0.5, tmp)
        compute_test(dtSq)

        project_boundary(positions)

        num_iter += 1
    # print("PBF iter: ", num_iter)

    epilogue(dt)


def run_pcg(dt):
    # Solve (M + k dt^2 J^T J) Δx = M(y - x) - k dt^2 J^T c

    old_positions.copy_from(positions)
    advect_velocity(dt)
    velocities.copy_from(velocities_adv)
    add(positions_adv, positions, dt, velocities)  # y = x + dt * v_adv

    positions.copy_from(positions_adv)
    project_boundary(positions)
    neighbor_search(positions)

    compute_density_and_Aii(positions)        # compute ρ, ∇W, Aii

    dtSq = dt * dt
    k = 1e-9

    build_block_diag_and_JtC(k, dtSq)         # build J^T C and diag(J^T J)
    build_rhs(k, dtSq)                        # rhs = M(y - x) - k dt^2 J^T c


    dx.fill(0.0)
    # r_vec.copy_from(rhs_vec)
    # apply_Minv(z_vec, r_vec)
    # p_vec.copy_from(z_vec)
    # rz_old = dot(r_vec, z_vec)


    compute_J_x(Ax, dx)                       # J * x
    compute_J_tr_x(Ap_vec, Ax)                # J^T (J x)
    finalize_A(Ap_vec, dx, Ap_vec, k, dtSq)   # A x = Mx + k dt^2 J^T J x

    apply_Minv(dx, rhs_vec)                   # Δx ≈ M^{-1} rhs
    # add(positions, positions, -1.0, dx)       # x ← x - Δx

    # for _ in range(max_jacobi_iter):
    #     compute_J_x(Ax, p_vec)
    #     compute_J_tr_x(Ap_vec, Ax)
    #     finalize_A(Ap_vec, p_vec, Ap_vec, k, dtSq)

    #     pAp = dot(p_vec, Ap_vec)
    #     alpha = rz_old / (pAp + 1e-20)

    #     add(dx, dx, alpha, p_vec)
    #     add(r_vec, r_vec, -alpha, Ap_vec)
    #     apply_Minv(z_vec, r_vec)

    #     rz_new = dot(r_vec, z_vec)
    #     beta = rz_new / (rz_old + 1e-20)
    #     add(p_vec, z_vec, beta, p_vec)
    #     rz_old = rz_new

    add(positions, positions, -1.0, dx)
    project_boundary(positions)
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
    canvas.circles(positions_normalized, radius=particle_radius_in_world, 
                  color=((particle_color >> 16 & 0xff) / 255.0,
                         (particle_color >> 8 & 0xff) / 255.0,
                         (particle_color & 0xff) / 255.0))
    
    # Render boundary particles in gray
    canvas.circles(boundary_positions_normalized, radius=0.004, 
                  color=(0.5, 0.5, 0.5))


@ti.kernel
def init_particles():

    alpha = (0.4 / cubic_value(0.0, h_)) 
    for i in range(num_particles):
        delta = h_ * 0.5
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5, boundary[1] * 0.06])
        positions[i] = ti.Vector([i % num_particles_x, i // num_particles_x]) * delta + offs
        for c in ti.static(range(dim)):
            velocities[i][c] = 0.0
        
        particle_is_dynamic[i] = 1 # 1 for fluid, 0 for boundary
        
        # Initialize per-particle rest density
        # You can customize this based on particle properties
        rho0[i] = 1000.0  # Default rest density
        
        # Set mass based on per-particle rest density
        m[i] = rho0[i] * alpha

    m_b.fill(1e+7)

    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

@ti.kernel 
def set_particle_rest_densities():
    """Set different rest densities for different particle regions or types"""
    for i in range(num_particles):
        x = i % num_particles_x
        y = i // num_particles_x
        
        # Example: Create density variation based on particle position
        # You can modify this to create different fluid types or density gradients
        if y < 20:  # Bottom particles have higher density
            rho0[i] = 1200.0
        elif y < 40:  # Middle particles have medium density  
            rho0[i] = 1000.0
        else:  # Top particles have lower density
            rho0[i] = 800.0
        
        # Update mass accordingly
        alpha = (0.4 / cubic_value(0.0, h_))
        m[i] = rho0[i] * alpha



def init_boundary_particles():
    boundary_positions.from_numpy(boundary_positions_np)
    # boundary_positions = compute_boundary_positions()

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
    set_particle_rest_densities()  # Set varied rest densities
    p.fill(0.0)


def stop():
    return False


def show_options(gui, frame_cnt):
    global boundary_particle_spacing, boundary_layers, use_measure_error, time_delta, tol, max_jacobi_iter, solver_method
    with gui.sub_window("Settings", 0., 0., 0.4, 0.4):
        gui.text(f"Current frame: {frame_cnt}")
        gui.text("")  # Spacer
        
        gui.text("Method:")
        solver_method = gui.slider_int("a", solver_method, 0, 3)
        if solver_method == 0:
            gui.text("Current: IISPH")
        elif solver_method == 1:
            gui.text("Current: PBF")
        elif solver_method == 2:
            gui.text("Current: PCG")
        elif solver_method == 3:
            gui.text("Current: IISPH(vanilla)")

        gui.text("Time Step:")
        time_delta = gui.slider_float("b", time_delta, 0.001, 0.1)
        
        gui.text("")  # Spacer
        gui.text("Max Iteration:")
        max_jacobi_iter = gui.slider_int("c", max_jacobi_iter, 1, 2000)
        
        gui.text("")  # Spacer
        gui.text("Tolerance:")
        tol = gui.slider_float(" ", tol, 1e-6, 1e-1)
        
        # gui.text("")  # Spacer
        # gui.text("Termination Condition:")
        # use_measure_error = gui.checkbox("Use measure_error()", use_measure_error)
        # if not use_measure_error:
        #     gui.text("Using residual norm")
        
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
    set_particle_rest_densities()  # Set varied rest densities
    init_boundary_particles()
    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")
    print(f"num_boundary_particles={num_boundary_particles}")
    print(f"boundary_layers={boundary_layers}")
    
    window = ti.ui.Window('PBF2D', screen_res, show_window=True, vsync=False)
    gui = window.get_gui()
    
    runSim = False
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

            if solver_method == 0:
                run_iisph(time_delta)
            elif solver_method == 1:
                run_pbf(time_delta)
            elif solver_method == 2:
                run_pcg(time_delta)
            elif solver_method == 3:
                run_iisph_vanilla(time_delta)


            # run_pbf(time_delta)
            frame_cnt += 1
            
        # if frame_cnt % 20 == 1:
        #     print_stats()
        
        render(window)
        window.show()

        
if __name__ == "__main__":
    main()