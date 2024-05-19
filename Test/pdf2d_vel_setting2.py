
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'Times New Roman'

import taichi as ti

ti.init(arch=ti.gpu)

screen_res = (600, 600)
screen_to_world_ratio = 10.0
boundary = (
    screen_res[0] / screen_to_world_ratio,
    screen_res[1] / screen_to_world_ratio,
)
cell_size = 2.51
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

density_error_avg = ti.field(dtype = ti.float32,shape=1)

grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))
dim = 2
bg_color = 0x112F41
particle_color = 0x068587
boundary_color = 0xEBACA2
num_particles_x = 100
num_particles = num_particles_x * 100
num_frames = 1000
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 60.0

epsilon = 1e-5
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_res[1]
per_vertex_color = ti.Vector.field(3, ti.float32, shape=num_particles)
indices = np.zeros(num_particles)
# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 1
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

b = ti.Vector.field(2, dtype=ti.f32, shape=1)
b[0] = ti.math.vec2(1.0, 1.0)
old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
positions_render = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
density_error_frames = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# velocities_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities, positions_render)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
ti.root.dense(ti.i, num_particles).place(density_error_frames)
# ti.root.dense(ti.i, num_particles).place(lambdas, velocities_deltas)
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
    vel_strength = 10.0
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
        g = ti.Vector([0.0, -9.81])
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
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0
        if density_constraint < 0:
            density_constraint = 0.

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j) * spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0
        position_deltas[p_i] = pos_delta_i
    # apply position deltas
    for i in positions:
        positions[i] += position_deltas[i]

    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)

        # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta
    #
    for p_i in positions:
        pos_i = positions[p_i]
        vel_i = velocities[p_i]
        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        velocity_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            vel_ji = vel_i - velocities[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            velocity_constraint += grad_j.dot(vel_ji)

        # Eq(1)
        velocity_constraint = (mass * velocity_constraint)

        if velocity_constraint < 0:
            velocity_constraint = 0.0

        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-velocity_constraint) / (sum_gradient_sqr + lambda_epsilon)
        # compute position deltas
        # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        vel_delta_i = ti.Vector([0.0, 0.0])
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            vel_delta_i += (lambda_i + lambda_j) * spiky_gradient(pos_ji, h_)

        # vel_delta_i /= rho0
        position_deltas[p_i] = vel_delta_i
        # apply position deltas
    for i in positions:
        velocities[i] += position_deltas[i]
        positions[i] = old_positions[i] + velocities[i] * time_delta



@ti.kernel
def epilogue():
    for i in positions:
        positions_render[i].x = positions[i].x * (screen_to_world_ratio / screen_res[0])
        positions_render[i].y = positions[i].y * (screen_to_world_ratio / screen_res[1])

    for p_i in positions:
        pos_i = positions[p_i]

        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_error_frames[p_i] = (mass * density_constraint / rho0) - 1
    # no vorticity/xsph because we cannot do cross product in 2D...

    # for i in velocities:
    #     vel_norm_i = ti.math.length(velocities[i])
    #
    #     if vel_norm_i <=
    #     (0.098, 0.51, 0.77)
    #     (0.13, 0.65, 0.94)
    #     (0.38, 0.74, 0.94)
    #     (0.65, 0.83, 0.92)
    #     (0.88, 0.88, 0.88)
    #
    #
    #
    # # ["#1984c5", "#22a7f0", "#63bff0", "#a7d5ed", "#e2e2e2", "#e1a692", "#de6e56", "#e14b31", "#c23728"]
    #
    # per_vertex_color.fill(ti.math.vec3(6, 133, 135))

def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()


def render(gui):
    gui.clear(bg_color)
    pos_np = positions.to_numpy()
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    # gui.circles(pos_np, radius=particle_radius, color=particle_color)

    gui.circles(pos_np, radius=particle_radius, palette=[particle_color, particle_color], palette_indices=indices)
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

#
# @ti.kernel
# def compute_heat_map_rgb():
#
#     arr = velocities.to_numpy()

@ti.kernel
def get_density_error() :
    for i in density_error_frames :
        density_error_avg[0] += density_error_frames[i]/num_particles


def main():
    frame=0
    init_particles()
    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")
    window = ti.ui.Window(name="PBF 2D(Velocity Adjustment)", res=screen_res)
    canvas = window.get_canvas()
    canvas.set_background_color((0.066, 0.18, 0.25))
    # scene = ti.ui.Scene()
    # camera = ti.ui.Camera()
    # camera.position()

    # for plotting
    # x_arr = []
    # frames = 0
    # while frames < num_frames:
    #     run_pbf()
    #     arr = density_error_frames.to_numpy()
    #     arr = np.absolute(arr)
    #     # print(arr.sum())
    #     x_arr.append(arr.sum() / num_particles)
    #     frames+=1
    #
    # matplotlib.use('TkAgg')
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.plot(x_arr)
    #
    # plt.xlabel('frames')
    # plt.ylabel('avg. density error')
    # plt.show()


    while window.running:
        if frame < 640 :
            move_board()

        run_pbf()
        arr = velocities.to_numpy()
        magnitudes = np.linalg.norm(arr, axis=1)  # Compute magnitudes of vectors
        norm = Normalize(vmin=np.min(magnitudes), vmax=np.max(magnitudes))
        heatmap_rgb = plt.cm.coolwarm(norm(magnitudes))[:, :3]  # Use plasma colormap for heatmap
        per_vertex_color.from_numpy(heatmap_rgb)
        # canvas.circles()
        canvas.circles(centers=positions_render, radius=particle_radius_in_world, per_vertex_color=per_vertex_color)
        # if i < 1000:
        # filename = f'results/frame_{i:05d}.png'  # create filename with suffix png
        # print(f'Frame {i} is recorded in {filename}')
        # arr = window.get_image_buffer_as_numpy()
        # # video_manager.write_frame(arr)
        # ti.tools.imwrite(arr, filename)
        # window.save_image(filename)

        if frame % 10 == 0:
            real_frame = frame // 10

            density_error_avg[0] = 0
            get_density_error()
            d_err = round(density_error_avg[0], 5)
            d_err = d_err if d_err > 0.0 else 0.0
            print(real_frame, " | ", d_err)
            # window.save_image("./results/pbf2d_vel_setting2.py/" + str(real_frame) + ".jpg")

        window.show()
        frame = frame+1

        # else:
        #     # video_manager.make_video(gif=True, mp4=True)
        #     # video_manager.get_output_filename(".mp4")
        #     window.destroy()

if __name__ == "__main__":
    main()