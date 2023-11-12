import taichi as ti
import numpy as np
import os
from tqdm import tqdm

import my_mesh
from my_mesh import Mesh
from my_solver import Solver

ti.init(kernel_profiler=True, arch=ti.cuda, device_memory_GB=6)
vec = ti.math.vec3

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window
dt = 0.003  # Larger dt might lead to unstable results.

per_vertex_color = ti.Vector.field(3, ti.float32, shape=4)
debug_edge_indices = ti.field(dtype=ti.i32, shape=2)

center = ti.Vector.field(n=3, dtype=ti.float32, shape=1)
center[0] = ti.math.vec3(0.5, 0.5, 0.5)
debug_edge_indices[0] = 0
debug_edge_indices[1] = 1
static_mesh_path = "seq_models/Kyra_DVStandClubbing/"
static_mesh_file = "Kyra_DVStandClubbing_" + str(0).zfill(4) + ".obj"
total_frame_num = 1


mesh = Mesh("obj_models/poncho_8K.obj", scale=0.3, trans=ti.math.vec3(0.5, 0.8, 0.5), rot=ti.math.vec3(0.0, 0.0, 0.0))
# mesh = Mesh("obj_models/clubbing_dress.obj", scale=0.8, trans=ti.math.vec3(0.5, -0.8, 0.5), rot=ti.math.vec3(90.0, 0.0, 0.0))
# mesh = Mesh("obj_models/square_16K.obj", scale=0.1, trans=ti.math.vec3(0.5, 0.8, 0.5), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh = Mesh("obj_models/square_16K.obj", scale=0.2, trans=ti.math.vec3(0.5, 0.8, 0.5), rot=ti.math.vec3(0.0, 0.0, 0.0))
# static_mesh = Mesh("obj_models/sphere5K.obj", scale=0.5, trans=ti.math.vec3(0.5, 0.5, 0.5), rot=ti.math.vec3(0.0, 0.0, 0.0))
static_mesh = Mesh(static_mesh_path + static_mesh_file, scale=0.8, trans=ti.math.vec3(0.5, -0.8, 0.5), rot=ti.math.vec3(90.0, 0.0, 0.0))

use_single_static_mesh = True

if not use_single_static_mesh:
    total_frame_num = len(os.listdir(static_mesh_path))
    static_meshes_pos_np = np.zeros((total_frame_num, static_mesh.mesh.verts.x.shape[0], 3))
    static_meshes_pos_np[0] = static_mesh.mesh.verts.x.to_numpy()

    def read_verts_only(path):
        with open(path, 'r') as f:
            lines = f.readlines()
            verts = []
            for line in lines:
                if line.startswith('v '):
                    line = line.strip().split()
                    vert = [float(line[1]), float(line[2]), float(line[3])]
                    verts.append(vert)

        verts_np = np.array(verts)
        return verts_np

    progress = tqdm(np.arange(total_frame_num))
    for f in progress:
        static_mesh_file = "Kyra_DVStandClubbing_" + str(f).zfill(4) + ".obj"
        new_verts_static = read_verts_only(static_mesh_path + static_mesh_file)
        static_meshes_pos_np[f] = new_verts_static
        progress.update(1)
        progress.set_description("Loading static meshes")
    print('Loading done:', static_meshes_pos_np.shape)
else:
    static_meshes_pos_np = np.zeros((1, static_mesh.mesh.verts.x.shape[0], 3))
    static_meshes_pos_np[0] = static_mesh.mesh.verts.x.to_numpy()

total_verts_np = mesh.mesh.verts.x.to_numpy()
total_verts_np = np.append(total_verts_np, static_mesh.mesh.verts.x.to_numpy(), axis=0)
object_range = np.max(total_verts_np, axis=0) - np.min(total_verts_np, axis=0)

total_min = ti.field(dtype=ti.f32, shape=(3,))
total_min.from_numpy(np.min(total_verts_np, axis=0) - object_range * 0.8)
total_max = ti.field(dtype=ti.f32, shape=(3,))
total_max.from_numpy(np.max(total_verts_np, axis=0) + object_range * 0.8)

static_meshes_pos_ti = ti.field(ti.math.vec3, shape=(total_frame_num, static_mesh.mesh.verts.x.shape[0]))
static_meshes_pos_ti.from_numpy(static_meshes_pos_np)

@ti.kernel
def init_color():
    per_vertex_color[0] = ti.math.vec3(1, 0, 0)
    per_vertex_color[1] = ti.math.vec3(0, 1, 0)
    per_vertex_color[2] = ti.math.vec3(0, 0, 1)
    per_vertex_color[3] = ti.math.vec3(0, 1, 1)

init_color()

sim = Solver(mesh, static_mesh=static_mesh, static_meshes=static_meshes_pos_ti, dt=dt, max_iter=1)

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 768), fps_limit=200)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1., 2.0, 3.5)
camera.fov(30)
camera.up(0, 1, 0)

run_sim = True
frame = 0
frame_rate = 40
while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            if not use_single_static_mesh:
                sim.update_static_mesh(frame=0, frame_rate=frame_rate, scale=1.0, trans=ti.math.vec3(0.5, -0.8, 0.5), rot=ti.math.vec3(90.0, 0.0, 0.0))
            sim.reset()
            frame = 0
            run_sim = False

    if run_sim:
        if not use_single_static_mesh and frame < frame_rate * total_frame_num:
            sim.update_static_mesh(frame=frame, frame_rate=frame_rate, scale=1.0, trans=ti.math.vec3(0.5, -0.8, 0.5), rot=ti.math.vec3(90.0, 0.0, 0.0))
        sim.update(dt=dt, num_sub_steps=6)
        # print('frame:', frame)
        frame += 1
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    camera.lookat(0.5, 0.5, 0.5)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))
    # scene.particles(sim.verts.x, radius=sim.radius, color=(1, 0.5, 0))
    # scene.lines()

    scene.mesh(static_mesh.mesh.verts.x, indices=static_mesh.face_indices)
    scene.mesh(sim.verts.x, indices=mesh.face_indices, color=(1, 0.5, 0))
    # scene.particles(static_mesh.mesh.verts.x, radius=sim.radius, color=(0, 1, 0))
    # scene.particles(static_mesh.mesh.edges.x, radius=sim.radius, color=(1, 0, 0))
    canvas.scene(scene)
    window.show()
    # window.save_image(f'results/test/{step:06d}.jpg')
