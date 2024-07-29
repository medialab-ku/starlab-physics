import taichi as ti
import json

from Scenes import test_ee as scene1
import os
from framework.physics import XPBD_unit_test as XPBD_unit_test
from framework.utilities import selection_tool as st

sim = XPBD_unit_test.Solver(g=ti.math.vec3(0.0, -9.81, 0.0), dt=0.03, stiffness_stretch=5e5, stiffness_bending=5e5, dHat=4e-3)
window = ti.ui.Window("PBD framework", (1024, 768), fps_limit=200)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((1., 1., 1.))
scene = window.get_scene()
camera = ti.ui.Camera()

init_x = 2.0
init_y = 4.0
init_z = 7.0
camera.position(init_x, init_y, init_z)
camera.fov(40)
camera.up(0, 1, 0)

run_sim = False
MODE_WIREFRAME = False
LOOKAt_ORIGIN = True

n_substep = 20
frame_end = 100

dt_ui = sim.dt
dHat_ui = sim.dHat

damping_ui = sim.damping

YM_ui = sim.stiffness_bending
YM_b_ui = sim.stiffness_stretch

friction_coeff_ui = sim.mu

mesh_export = False
frame_cpu = 0

def show_options():

    global n_substep
    global dt_ui
    global damping_ui
    global YM_ui
    global YM_b_ui
    global sim
    global dHat_ui
    global friction_coeff_ui
    global MODE_WIREFRAME
    global LOOKAt_ORIGIN
    global mesh_export
    global frame_end

    old_dt = dt_ui
    old_dHat = dHat_ui
    old_friction_coeff = dHat_ui
    old_damping = damping_ui
    YM_old = YM_ui
    YM_b_old = YM_b_ui

    with gui.sub_window("XPBD Settings", 0., 0., 0.3, 0.7) as w:

        dt_ui = w.slider_float("dt", dt_ui, 0.001, 0.101)
        n_substep = w.slider_int("# sub", n_substep, 1, 100)
        dHat_ui = w.slider_float("dHat", dHat_ui, 0.0001, 0.0301)
        friction_coeff_ui = w.slider_float("fric. coef.", friction_coeff_ui, 0.0, 1.0)
        damping_ui = w.slider_float("damping", damping_ui, 0.0, 1.0)
        YM_ui = w.slider_float("YM", YM_ui, 0.0, 1e8)
        YM_b_ui = w.slider_float("YM_b", YM_b_ui, 0.0, 1e8)

        frame_str = "# frame: " + str(frame_cpu)
        w.text(frame_str)

        LOOKAt_ORIGIN = w.checkbox("Look at origin", LOOKAt_ORIGIN)
        sim.enable_velocity_update = w.checkbox("velocity constraint", sim.enable_velocity_update)
        sim.enable_collision_handling = w.checkbox("handle collisions", sim.enable_collision_handling)
        mesh_export = w.checkbox("export mesh", mesh_export)

        if mesh_export is True:
            frame_end = w.slider_int("end frame", frame_end, 1, 2000)

        # w.text("")
        # w.text("dynamic mesh stats.")
        # verts_str = "# verts: " + str(sim.max_num_verts_dy)
        # edges_str = "# edges: " + str(sim.max_num_edges_dy)
        # faces_str = "# faces: " + str(sim.max_num_faces_dy)
        # w.text(verts_str)
        # w.text(edges_str)
        # w.text(faces_str)
        # w.text("")
        # w.text("static mesh stats.")
        # verts_str = "# verts: " + str(sim.max_num_verts_st)
        # edges_str = "# edges: " + str(sim.max_num_edges_st)
        # faces_str = "# faces: " + str(sim.max_num_faces_st)
        # w.text(verts_str)
        # w.text(edges_str)
        # w.text(faces_str)
        #

    if not old_dt == dt_ui:
        sim.dt = dt_ui

    if not old_dHat == dHat_ui:
        sim.dHat = dHat_ui

    if not old_friction_coeff == friction_coeff_ui:
        sim.mu = friction_coeff_ui

    if not YM_old == YM_ui:
        sim.stiffness_bending = YM_ui

    if not YM_b_old == YM_b_ui:
        sim.stiffness_stretch = YM_b_ui

    if not old_damping == damping_ui:
        sim.damping = damping_ui

def load_animation():
    global sim

    with open('framework/animation/animation.json') as f:
        animation_raw = json.load(f)
    animation_raw = {int(k): v for k, v in animation_raw.items()}

    animationDict = {(i+1):[] for i in range(4)}

    for i in range(4):
        ic = i + 1
        icAnimation = animation_raw[ic]
        listLen = len(icAnimation)
        # print(listLen)
        assert listLen % 7 == 0,str(ic)+"th Animation SETTING ERROR!! ======"

        num_animation = listLen // 7

        for a in range(num_animation) :
            animationFrag = [animation_raw[ic][k + 7*a] for k in range(7)] # [vx,vy,vz,rx,ry,rz,frame]
            animationDict[ic].append(animationFrag)

while window.running:

    if LOOKAt_ORIGIN:
        camera.lookat(0.0, 0.0, 0.0)

    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))

    if window.get_event(ti.ui.PRESS):
        # if window.event.key == 'u':
        #     g_selector.remove_all_sewing()


        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            frame_cpu = 0
            camera.position(init_x, init_y, init_z)
            sim.reset()
            run_sim = False

        if window.event.key == 'v':
            sim.enable_velocity_update = not sim.enable_velocity_update
            if sim.enable_velocity_update is True:
                print("velocity update on")
            else:
                print("velocity update off")

        if window.event.key == 'z':
            sim.enable_collision_handling = not sim.enable_collision_handling
            if sim.enable_collision_handling is True:
                print("collision handling on")
            else:
                print("collision handling off")
    if run_sim:
        # sim.animate_handle(g_selector.is_selected)
        sim.forward(n_substeps=n_substep)
        frame_cpu += 1

    show_options()

    scene.particles(sim.x_test, radius=0.05, per_vertex_color=sim.color_test)
    scene.mesh(sim.x_test, indices=sim.face_indices_test, color=(0.0, 0.0, 0.0), show_wireframe=True)

    scene.particles(sim.x_test_st, radius=0.05, color=(0.5, 0.5, 0.5))
    scene.mesh(sim.x_test_st, indices=sim.face_indices_test, color=(0.0, 0.0, 0.0), show_wireframe=True)

    camera.track_user_inputs(window, movement_speed=0.8, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()
