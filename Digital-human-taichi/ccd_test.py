import taichi as ti
import numpy as np
import os
from tqdm import tqdm

import my_mesh
from my_mesh import Mesh
from my_solver import Solver

ti.init(arch=ti.cuda, device_memory_GB=6)
vec = ti.math.vec3

window_size = 1024  # Number of pixels of the window
dt = 0.001 # Larger dt might lead to unstable results.

total_frame_num = 1
s_scale = 0.8
s_trans = ti.math.vec3(0.5, -0.8, 0.5)
s_rot = ti.math.vec3(00.0, 0.0, 0.0)

mesh = Mesh("obj_models/cube.obj", scale=0.3, trans=ti.math.vec3(0.5, 0.8, 0.5), rot=ti.math.vec3(0.0, 0.0, 0.0))
static_mesh = Mesh("obj_models/cube.obj", scale=0.3, trans=ti.math.vec3(0.5, 0.4, 0.5), rot=ti.math.vec3(0.0, 0.0, 0.0))


sim = Solver(mesh, static_mesh=static_mesh, static_meshes=None, dt=dt, max_iter=10)

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 768), fps_limit=200)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1., 2.0, 3.5)
camera.fov(30)
camera.up(0, 1, 0)
camera.lookat(0.5, 0.5, 0.5)


run_sim = True
frame = 0
frame_rate = 20
while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            sim.reset()
            frame = 0
            run_sim = False

        dx = 50
        if window.event.key == ti.ui.RIGHT:
            sim.static_mesh_move(0, dx)
        if window.event.key == ti.ui.LEFT:
            sim.static_mesh_move(1, dx)
        if window.event.key == ti.ui.UP:
            sim.static_mesh_move(2, dx)
        if window.event.key == ti.ui.DOWN:
            sim.static_mesh_move(3, dx)

    if run_sim:
        sim.update(dt=dt, num_sub_steps=1)
        # print('frame:', frame)
        frame += 1

    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)

    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))
    # scene.particles(sim.verts.x, radius=sim.radius, color=(1, 0.5, 0))
    # scene.lines()

    scene.mesh(static_mesh.mesh.verts.x, indices=static_mesh.face_indices, show_wireframe=False)
    # scene.mesh(static_mesh2.mesh.verts.x, indices=static_mesh2.face_indices, show_wireframe=True)
    # scene.lines(static_mesh.mesh.verts.x, indices=static_mesh.edge_indices, width=0.5,  color=(0, 0, 0))
    scene.mesh(sim.verts.x, indices=mesh.face_indices, color=(1, 0.5, 0), show_wireframe=False)
    # scene.lines(sim.verts.x, indices=mesh.edge_indices, width=0.5,  color=(0, 0, 0))
    if sim.button.shape[0] > 0:
        # with gui.sub_window("Sub window", x=0, y=0, width=0.1, height=0.1):
        #     gui.text("Button")
        #     gui.text("Position: " + str(sim.button[0]))
        #     gui.text("Size: " + str(sim.b_size[0]))
        scene.particles(sim.button, radius=sim.b_size[0], color=(0, 0, 1))

    canvas.scene(scene)
    window.show()