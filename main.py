import os
import argparse
import taichi as ti
import numpy as np
import time
import trimesh as tm
from config_builder import SimConfig
from scene_loader import SceneLoader
from particle_system import ParticleSystem
from framework import Framework
from neighbour_search import NeighborSearch
from pressure import Pressure
from surface_tension import SurfaceTension
from viscosity import Viscosity
from elasticity import Elasticity
from visualization import VisualizationEngine, VisualizationSettings, ColorMode, HeatmapField

from cache_system import SimulationCache
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap


ti.init(arch=ti.gpu, device_memory_fraction=0.7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)

    scene_name = scene_path.split("/")[-1].split(".")[0]
    # Per-run PLY/OBJ output directory: output/<scene>/<timestamp>
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    ply_out_dir = os.path.join("output", scene_name, timestamp_str)
    obj_out_dir = os.path.join("output", scene_name, timestamp_str, "mesh_obj")

    substeps = config.get_cfg("numSubstepping")
    # print(substeps)
    output_frames = config.get_cfg("exportFrame")
    output_interval = 20
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    series_prefix = os.path.join(ply_out_dir, "particle_object_{}.ply")
    if output_frames:
        os.makedirs(f"{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(ply_out_dir, exist_ok=True)
    if output_obj:
        os.makedirs(obj_out_dir, exist_ok=True)

    # method = config.get_cfg("simulationMethod")
    loader = SceneLoader(config)
    scene_data = loader.prepare_scene()
    ps = ParticleSystem(config, GGUI=True)
    loader.populate_scene(ps, scene_data)
    loader.reset_emitter_system()

    ns = NeighborSearch(config, ps)
    pressure = Pressure(ps)
    viscosity = Viscosity(ps)
    surface_tension = SurfaceTension(ps)
    elasticity = Elasticity(ps)
    fw = Framework(ps, ns, pressure, viscosity, surface_tension, elasticity)

    # fw.initialize()

    window = ti.ui.Window('SPH', (1024, 1024), show_window=True, vsync=False)
    gui = window.get_gui()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(5.5, 2.5, -4.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(0.0, 0.0, 1.0)
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Invisible objects
    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []

    # Transparent objects
    transparent_objects = config.get_cfg("transparentObjects")
    if not transparent_objects:
        transparent_objects = []
    color_alpha = config.get_cfg("alpha")
    if color_alpha is None:
        color_alpha = 1.0

    # Invisible flag for transparent objects
    is_invisible = False

    # # Visualization mode
    # viz_mode = 1  # 1: heatmap, 2: original colors
    # heatmap_type = 2  # 1: velocity, 2: divergence, 3: density

    viz_settings = VisualizationSettings(
        color_mode=ColorMode.heatmap,
        heatmap_field={1: HeatmapField.velocity, 2: HeatmapField.divergence, 3: HeatmapField.density}[3],
        transparent_objects=transparent_objects,
        invisible_objects=invisible_objects,
        color_alpha=color_alpha,
    )
    viz = VisualizationEngine(ps, config, viz_settings)
    # Export options
    export_rigid_objects = False
    export_rigid_mesh = bool(output_obj)
    export_stats = False


    frame_cnt = 0
    export_ply = output_ply
    end_frame = 8000

    # Caching system
    cache = SimulationCache(ps, fw, max_steps=10)
    cache.snapshot_baseline(anim_time=0.0)


    def show_options_solver():
        with gui.sub_window("Solver settings", 0., 0., 0.4, 0.3) as w:
            fw.dt = w.slider_float("dt", fw.dt, 0.001, 0.04)
            # fw.cfl = w.checkbox("CFL", fw.cfl)
            # fw.num_substep = w.slider_int("substepping", fw.num_substep, 1, 100)

            # if method == 2:
            # fw.tol_opt = w.slider_int("opt tol magnitude", fw.tol_opt, 1, 7)
            # fw.max_iteration_opt = w.slider_int("max opt. iter", fw.max_iteration_opt, 1, 1000)
            #
            # fw.density_error = w.checkbox("density error", fw.density_error)
            # # fw.iisph = w.checkbox("iisph", fw.iisph)
            #
            # # if fw.iisph:
            # #     fw.omega = w.slider_float("relaxation", fw.omega, 0.001, 2.0)
            #
            # # else:
            #     # fw.smooth_max = w.checkbox("smooth max(Ours)", fw.smooth_max)
            #     # if fw.smooth_max:
            # fw.eps = w.slider_float("eps", fw.eps, 0.001, 10.0)
            # fw.max_iteration_pcg = w.slider_int("max pcg. iter", fw.max_iteration_pcg, 1, 1000)
            # fw.tol_pcg = w.slider_int("pcg tol magnitude", fw.tol_pcg, 1, 15)
            # fw.use_pcg = w.checkbox("warm-start", fw.use_pcg)
            #
            # fw.enable_DF = w.checkbox("DF solve", fw.enable_DF)

            try:
                N_active = int(ps.particle_num[None])
                mats = ps.material.to_numpy()[:N_active]
                fluid_cnt = int((mats == ps.material_fluid).sum())
            except Exception:
                fluid_cnt = int(ps.fluid_particle_num)
            gui.text(f"# fluid particle: {fluid_cnt}")
            gui.text(f"# boundary particle: {ps.rigid_particle_num}")
            gui.text(f"Current frame: {frame_cnt}")


    def show_options_visual():
        global export_rigid_objects
        global export_ply
        global export_stats
        global export_rigid_mesh
        global end_frame

        with gui.sub_window("Visualization settings", 0.0, 0.4, 0.4, 0.3) as w:

            export_ply = w.checkbox("export particles (PLY)", export_ply)
            if export_ply:
                export_rigid_objects = w.checkbox("Export rigid particles", export_rigid_objects)
            export_rigid_mesh = w.checkbox("Export rigid mesh (OBJ)", export_rigid_mesh)
            if export_ply or export_rigid_mesh:
                end_frame = w.slider_int("end frame", end_frame, 0, int(3e4))
            export_stats = w.checkbox("export stats", export_stats)

            gui.text("")  # Spacer
            gui.text("Visualization Controls:")
            viz.render_ui(w, gui)


    cnt = 0
    cnt_ply = 0
    cnt_obj = 0
    runSim = False


    @ti.kernel
    def reset_R_identity(R: ti.template()):
        for i in ti.grouped(R):
            R[i] = ti.math.mat3([[1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])


    # --- helper: compute rigid transform (R, t) by Kabsch from particles ---
    def compute_rigid_transform_from_particles(obj_id: int):
        N = int(ps.particle_num[None])
        if N <= 0:
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            return R, t
        obj_ids = ps.object_id.to_numpy()[:N]
        mats = ps.material.to_numpy()[:N]
        mask = (obj_ids == obj_id) & (mats == ps.material_solid)
        if not np.any(mask):
            R = np.eye(3, dtype=np.float32)
            t = np.zeros(3, dtype=np.float32)
            return R, t
        X0 = ps.x_0.to_numpy()[:N][mask].astype(np.float32)
        X = ps.x.to_numpy()[:N][mask].astype(np.float32)
        c0 = X0.mean(axis=0)
        c = X.mean(axis=0)
        P = X0 - c0
        Q = X - c
        R = np.eye(3, dtype=np.float32)
        if P.shape[0] >= 3:
            H = P.T @ Q
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
        t = c - R @ c0
        return R.astype(np.float32), t.astype(np.float32)


    while window.running:

        show_options_solver()
        show_options_visual()
        # show_options_stats()
        # Cache UI
        res_cache = cache.show_ui(gui, current_frame=frame_cnt, pos=(0.4, 0.0), size=(0.3, 0.25))
        if res_cache.get("restored", False):
            runSim = False
            frame_cnt = int(res_cache.get("frame", frame_cnt))

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                # Toggle run state
                runSim = not runSim
                if runSim:
                    ps.x_old.copy_from(ps.x)
                    ps.v_adv.copy_from(ps.v)

            if window.event.key == 'b':
                # Rewind one cached frame
                runSim = False
                result = cache.rewind_one(frame_cnt)
                if result.get("restored", False):
                    frame_cnt = int(result.get("frame", frame_cnt))
                    print(f"rewind: {result.get('rewind_steps', 0)} frames")

            if window.event.key == 'r':
                print("reset simulation...")
                ok, _, _ = (False, None, None)
                ok, _, _ = cache.restore_baseline()

                if ok:
                    # Reset GUI counters and pause
                    frame_cnt = 0
                    cnt_ply = 0
                    runSim = False
                    loader.reset_emitter_system()

        if (export_ply or export_rigid_mesh) and frame_cnt > end_frame:
            runSim = False

        if runSim:
            dt_frame = fw.dt
            dt_sub = dt_frame
            try:
                if getattr(fw, "cfl", False) and hasattr(fw, "compute_cfl_dt"):
                    dt_sub = float(fw.compute_cfl_dt(dt_sub))
            except Exception:
                pass
            fw.dt = dt_sub
            for i in range(1):
                fw.current_frame = int(frame_cnt  + i)
                loader.step_emitter_system(fw.dt)
                fw.forward()

            fw.dt = dt_frame
            frame_cnt += 1

            # After completing a frame, cache the end-of-frame state
            cache.push(frame_cnt=frame_cnt)

            if frame_cnt > 0 and frame_cnt % output_interval == 0:
                if export_ply:
                    # Ensure per-run output directory exists
                    try:
                        os.makedirs(ply_out_dir, exist_ok=True)
                    except Exception:
                        pass
                    if export_rigid_objects:
                        # Export each object separately
                        for obj_id in ps.object_collection:
                            obj_data = ps.dump(obj_id=obj_id)
                            np_pos = obj_data["position"]

                            # Only export if object has particles
                            if len(np_pos) > 0:
                                if obj_id == 0:
                                    # Fluid particles (object id 0): position + vertex color encoding
                                    # R: |v| normalized, G: density (current heatmap scheme), B: 0
                                    np_vel = obj_data["velocity"]

                                    # Speed magnitude normalization (match viz setting)
                                    speed = np.linalg.norm(np_vel, axis=1)
                                    norm_speed = Normalize(vmin=0.0, vmax=1.5, clip=True)
                                    r_chan = norm_speed(speed)

                                    # Density difference normalization (match viz density scheme)
                                    N_active = int(ps.particle_num[None])
                                    object_id_np = ps.object_id.to_numpy()[:N_active]
                                    mask_fluid = (object_id_np == obj_id)
                                    density_np = ps.density.to_numpy()[:N_active]
                                    density0_np = ps.density0.to_numpy()[:N_active]
                                    dens_diff = (density_np - density0_np)[mask_fluid]
                                    norm_dens = Normalize(vmin=-50.0, vmax=50.0, clip=True)
                                    g_chan = norm_dens(dens_diff)

                                    b_chan = np.zeros_like(r_chan)

                                    # Create separate PLY file for fluid
                                    obj_series_prefix = os.path.join(ply_out_dir, f"particle_object_{obj_id}.ply")
                                    writer = ti.tools.PLYWriter(num_vertices=len(np_pos))
                                    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])

                                    # Map density heatmap colors (blue-white-red) directly to RGB
                                    cmap = LinearSegmentedColormap.from_list("heatmap", ["blue", "white", "red"])
                                    rgba_colors = cmap(g_chan)  # g_chan holds normalized density difference
                                    writer.add_vertex_color(rgba_colors[:, 0], rgba_colors[:, 1], rgba_colors[:, 2])

                                    writer.export_frame_ascii(cnt_ply, obj_series_prefix)
                                else:
                                    # Rigid objects (object id > 0): position only
                                    obj_series_prefix = os.path.join(ply_out_dir, f"particle_object_{obj_id}.ply")
                                    writer = ti.tools.PLYWriter(num_vertices=len(np_pos))
                                    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])

                                    # Add object ID as vertex color (R channel)
                                    object_id_color = np.full((len(np_pos), 3), 0.0)
                                    object_id_color[:, 0] = obj_id / 255.0  # Normalize object ID to 0-1 range
                                    writer.add_vertex_color(object_id_color[:, 0], object_id_color[:, 1],
                                                            object_id_color[:, 2])

                                    writer.export_frame_ascii(cnt_ply, obj_series_prefix)
                    else:
                        # Export only fluid particles (object id 0) with velocity encoding
                        obj_id = 0
                        obj_data = ps.dump(obj_id=obj_id)
                        np_pos = obj_data["position"]

                        # Only export if object has particles
                        if len(np_pos) > 0:
                            np_vel = obj_data["velocity"]

                            # R: |v| normalized, G: density (heatmap), B: 0
                            speed = np.linalg.norm(np_vel, axis=1)
                            norm_speed = Normalize(vmin=0.0, vmax=1.5, clip=True)
                            r_chan = norm_speed(speed)

                            N_active = int(ps.particle_num[None])
                            object_id_np = ps.object_id.to_numpy()[:N_active]
                            mask_fluid = (object_id_np == obj_id)
                            density_np = ps.density.to_numpy()[:N_active]
                            density0_np = ps.density0.to_numpy()[:N_active]
                            dens_diff = (density_np - density0_np)[mask_fluid]
                            norm_dens = Normalize(vmin=-50.0, vmax=50.0, clip=True)
                            g_chan = norm_dens(dens_diff)

                            # Use density heatmap RGB for fluid
                            cmap = LinearSegmentedColormap.from_list("heatmap", ["blue", "white", "red"])
                            rgba_colors = cmap(g_chan)

                            writer = ti.tools.PLYWriter(num_vertices=len(np_pos))
                            writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])

                            writer.add_vertex_color(rgba_colors[:, 0], rgba_colors[:, 1], rgba_colors[:, 2])

                            writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
                    cnt_ply += 1

                # Export rigid meshes (OBJ) using per-frame rigid transform
                if export_rigid_mesh and len(ps.object_id_rigid_body) > 0:
                    try:
                        os.makedirs(obj_out_dir, exist_ok=True)
                    except Exception:
                        pass
                    for r_body_id in ps.object_id_rigid_body:
                        try:
                            rb = ps.object_collection.get(r_body_id, None)
                            if rb is None:
                                continue
                            mesh_rest = rb.get("mesh", None)
                            if mesh_rest is None:
                                continue
                            Rm, tm_vec = compute_rigid_transform_from_particles(int(r_body_id))
                            V_rest = np.asarray(mesh_rest.vertices, dtype=np.float32)
                            V_tr = V_rest @ Rm.T + tm_vec[None, :]
                            F = np.asarray(mesh_rest.faces) if hasattr(mesh_rest, "faces") else None
                            mesh_out = tm.Trimesh(vertices=V_tr, faces=F, process=False)
                            out_path = os.path.join(obj_out_dir, f"obj_{int(r_body_id)}_{cnt_obj:06}.obj")
                            mesh_out.export(out_path)
                        except Exception:
                            pass
                    cnt_obj += 1

        viz.update_buffers()

        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius, color=particle_color)
        else:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
            scene.set_camera(camera)
            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            viz.draw(scene, canvas, background_color=background_color)

        cnt += 1
        # if cnt > 6000:
        #     break
        window.show()

