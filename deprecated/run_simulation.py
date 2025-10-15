import os
import argparse
import taichi as ti
import numpy as np
import time
import trimesh as tm
from config_builder import SimConfig
from particle_system import ParticleSystem
from animation import AnimationSystem
from cache_system import SimulationCache
import matplotlib.pyplot as plt
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
    ply_out_dir = os.path.join("../output", scene_name, timestamp_str)
    obj_out_dir = os.path.join("../output", scene_name, timestamp_str, "mesh_obj")

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

    method = config.get_cfg("simulationMethod")
    ps = ParticleSystem(config, GGUI=True)
    solver = ps.build_solver()
    solver.initialize()

    # # Add a kernel for moving the boundary object
    # @ti.kernel
    # def move_boundary_object(dt: float):
    #     amplitude = -0.3
    #     frequency = 4.0
    
    #     for p_i in ti.grouped(ps.x):
    #         if ps.object_id[p_i] == 2:
    #             # Update position based on the initial position x_0
    #             new_y = ti.cos(dt) * ps.x[p_i][1] + ti.sin(dt) * ps.v[p_i][1]
    #             new_v = -ti.sin(dt) * ps.x[p_i][1] + ti.cos(dt) * ps.v[p_i][1]
    #             ps.x[p_i][1] = new_y
    #             ps.v[p_i][1] = new_v

    window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)
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

    # Visualization mode
    viz_mode = 1  # 1: heatmap, 2: original colors
    heatmap_type = 2  # 1: velocity, 2: divergence, 3: density

    # Export options
    export_rigid_objects = False
    export_rigid_mesh = bool(output_obj)
    export_stats = False


    # Draw the lines for domain
    x_max, y_max, z_max = config.get_cfg("domainEnd")
    box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
    box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
    box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
    box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
    box_anchors[3] = ti.Vector([x_max, y_max, 0.0])

    box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
    box_anchors[5] = ti.Vector([0.0, y_max, z_max])
    box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
    box_anchors[7] = ti.Vector([x_max, y_max, z_max])

    box_lines_indices = ti.field(int, shape=(2 * 12))

    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val

    frame_cnt = 0
    export_ply = output_ply
    end_frame = 8000

    # Initialize animation system
    animator = AnimationSystem(ps, config)
    anim_time = 0.0
    anim_nudge = 0.1  # translation speed (units/sec)
    rot_nudge = 0.0872664626  # rotation speed (rad/sec ~5deg)
    runAnim = False  # auto animation play/pause state
    # Auto mode is true if any animation in config has auto=true
    anim_auto_mode = False
    try:
        anim_auto_mode = bool(animator.has_auto())
    except Exception:
        anim_auto_mode = False

    # Animation handled by AnimationSystem

    # Caching system
    cache = SimulationCache(ps, solver, max_steps=10)
    # Capture baseline snapshot for instant soft reset
    try:
        cache.snapshot_baseline(anim_time=0.0)
    except Exception:
        pass

    def show_options_solver():

        with gui.sub_window("Solver settings", 0., 0., 0.4, 0.3) as w:

            solver.dt = w.slider_float("dt", solver.dt, 0.001, 0.04)
            solver.cfl = w.checkbox("CFL", solver.cfl)
            solver.num_substep = w.slider_int("substepping", solver.num_substep, 1, 100)

            if method == 2:
                solver.tol_opt = w.slider_int("opt tol magnitude", solver.tol_opt, 1, 7)
                solver.max_iteration_opt = w.slider_int("max opt. iter", solver.max_iteration_opt, 1, 1000)

                solver.density_error = w.checkbox("density error", solver.density_error)
                solver.iisph = w.checkbox("iisph", solver.iisph)

                if solver.iisph:
                    solver.omega = w.slider_float("relaxation", solver.omega, 0.001, 2.0)

                else:
                    solver.smooth_max = w.checkbox("smooth max(Ours)", solver.smooth_max)
                    if solver.smooth_max:
                        solver.eps = w.slider_float("eps", solver.eps, 0.001, 10.0)
                        solver.max_iteration_pcg = w.slider_int("max pcg. iter", solver.max_iteration_pcg, 1, 1000)
                        solver.tol_pcg = w.slider_int("pcg tol magnitude", solver.tol_pcg, 1, 15)
                        
                        solver.use_pcg = w.checkbox("warm-start", solver.use_pcg)

                solver.enable_DF = w.checkbox("DF solve", solver.enable_DF)


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

        global end_frame
        global heatmap
        global color_alpha
        global is_invisible
        global viz_mode
        global heatmap_type
        global export_rigid_objects
        global export_ply
        global export_stats
        global export_rigid_mesh

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
            viz_mode = w.slider_int("visualization mode", viz_mode, 1, 2)
            if viz_mode == 1:
                gui.text("Heatmap Mode")
                heatmap_type = w.slider_int("heatmap type", heatmap_type, 1, 3)
                if heatmap_type == 1:
                    gui.text("Velocity")
                elif heatmap_type == 2:
                    gui.text("Divergence")
                elif heatmap_type == 3:
                    gui.text("Density")
            elif viz_mode == 2:
                gui.text("Original Colors Mode")
            #
            gui.text("")  # Spacer
            gui.text("Transparency Controls:")
            if transparent_objects:
                is_invisible = w.checkbox("invisible", is_invisible)
                if is_invisible:
                    color_alpha = 0.0
                else:
                    color_alpha = 0.2
                gui.text("")  # Spacer
                gui.text(f"Transparent objects: {transparent_objects}")
                gui.text(f"Current alpha: {color_alpha:.2f}")
            else:
                gui.text("No transparent objects configured")

            gui.text("")
            if anim_auto_mode:
                gui.text(f"Auto animation: {'playing' if runAnim else 'paused'}")
                gui.text("p=play/pause, o=reset")
            else:
                gui.text("Manual animation: key mapping by axis")
                gui.text("oscillate: y=Up/Down, x/z=Left/Right; rotate: y=Left/Right, x/z=Up/Down")

    def show_options_stats():
        with gui.sub_window("Stats. settings", 0.7, 0.0, 0.3, 0.25) as w:

            solver.print_opt_iter  = w.checkbox("print opt iter", solver.print_opt_iter)
            solver.print_opt_error = w.checkbox("print opt error", solver.print_opt_error)
            solver.print_pcg_iter  = w.checkbox("print pcg iter", solver.print_pcg_iter)
            solver.print_pcg_error = w.checkbox("print pcg error", solver.print_pcg_error)
            solver.print_elapsed_time = w.checkbox("print elapsed time", solver.print_elapsed_time)
            solver.print_kinetic_energy = w.checkbox("print kinetic energy", solver.print_kinetic_energy)

        
            

    # -----------------------------
    # Stats export helpers
    # -----------------------------
    def _clear_solver_stats_if_any():
        try:
            if hasattr(solver, "clear_stats"):
                solver.clear_stats()
        except Exception:
            pass

    def _export_solver_stats_if_any():
        try:
            if not export_stats:
                return
            if not hasattr(solver, "get_stats_numpy"):
                return
            stats = solver.get_stats_numpy()
            if not isinstance(stats, dict) or len(stats) == 0:
                return
            os.makedirs(os.path.join("../data", "stats"), exist_ok=True)
            # Build descriptive prefix instead of timestamp
            # Format: dt<dt>-tol<tol_opt>-opt<maxOptIter>(-cfl)(-warmstart)
            try:
                label = "ours" if bool(getattr(solver, "smooth_max", False)) else "2014Bender"
            except Exception:
                label = "unknown"
            try:
                dt_val = float(getattr(solver, "dt", 0.0))
            except Exception:
                dt_val = 0.0
            try:
                tol_opt_val = int(getattr(solver, "tol_opt", 0))
            except Exception:
                tol_opt_val = 0
            try:
                max_iter_opt_val = int(getattr(solver, "max_iteration_opt", 0))
            except Exception:
                max_iter_opt_val = 0
            try:
                use_pcg_flag = bool(getattr(solver, "use_pcg", False)) if label == "ours" else False
            except Exception:
                use_pcg_flag = False
            # CFL flag (bool)
            try:
                cfl_flag = bool(getattr(solver, "cfl", False))
            except Exception:
                cfl_flag = False
            # IISPH toggle (bool)
            try:
                iisph_flag = bool(getattr(solver, "iisph", False))
            except Exception:
                iisph_flag = False
            # Base prefix includes dt, tol, and max opt iter, and optionally -cfl
            prefix_parts = [f"dt{dt_val:.5f}", f"tol{tol_opt_val}", f"opt{max_iter_opt_val}"]
            if cfl_flag:
                prefix_parts.append("cfl")
            # Error metric tag (always include for clarity)
            try:
                density_error_flag = bool(getattr(solver, "density_error", False))
            except Exception:
                density_error_flag = False
            error_metric_tag = "errden" if (iisph_flag or density_error_flag) else "errl2"
            prefix_parts.append(error_metric_tag)
            # Add warmstart tag to prefix for our method when enabled
            if label == "ours" and use_pcg_flag:
                prefix_parts.append("warmstart")
            prefix = "-".join(prefix_parts)
            # Variant selection
            if iisph_flag:
                variant = "iisph"
            else:
                if label == "ours":
                    # Drop pcg/nopcg suffix; always use 'ours'
                    variant = "ours"
                else:
                    # 2014Bender has no warmstart option
                    variant = "2014Bender"
            for name, arr in stats.items():
                try:
                    out_path = os.path.join("../data", "stats", f"{scene_name}-{prefix}-{variant}-{name}.npy")
                    np.save(out_path, arr)
                except Exception:
                    pass
        except Exception:
            # Do not crash UI due to stats export errors
            pass

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
        X0 = ps.x0.to_numpy()[:N][mask].astype(np.float32)
        X  = ps.x.to_numpy()[:N][mask].astype(np.float32)
        c0 = X0.mean(axis=0)
        c  = X.mean(axis=0)
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
        show_options_stats()
        # Cache UI
        res_cache = cache.show_ui(gui, current_frame=frame_cnt, pos=(0.4, 0.0), size=(0.3, 0.25))
        if res_cache.get("restored", False):
            runSim = False
            try:
                anim_time = float(res_cache.get("anim_time", anim_time))
                frame_cnt = int(res_cache.get("frame", frame_cnt))
            except Exception:
                pass

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                # Toggle run state
                runSim = not runSim
                anim_auto_mode = bool(animator.has_auto())
                if runSim:
                    if anim_auto_mode:
                        runAnim = True
                    ps.x_old.copy_from(ps.x)
                    ps.v_adv.copy_from(ps.v)
                    
                    # Fresh session: reset per-session stats if supported
                    _clear_solver_stats_if_any()
                else:
                    # Stopped: export stats if requested
                    _export_solver_stats_if_any()

            if window.event.key == 'b':
                # Rewind one cached frame (if available)
                runSim = False
                result = cache.rewind_one(frame_cnt)
                if result.get("restored", False):
                    try:
                        anim_time = float(result.get("anim_time", anim_time))
                        frame_cnt = int(result.get("frame", frame_cnt))
                    except Exception:
                        pass
                    print(f"rewind: {result.get('rewind_steps', 0)} frames")

            if window.event.key == 'r':
                print("reset simulation...")
                # Before resetting, export current stats if requested
                _export_solver_stats_if_any()
                # Clear rolling cache but keep baseline
                try:
                    cache.clear()
                except Exception:
                    pass

                # Try baseline restore for soft reset
                ok, _, _ = (False, None, None)
                try:
                    ok, _, _ = cache.restore_baseline()
                except Exception:
                    ok = False

                if ok:
                    # Reset GUI counters and pause
                    frame_cnt = 0
                    cnt_ply = 0
                    runSim = False
                    anim_time = 0.0
                    # Reset toggled rigid bodies to static state
                    try:
                        ps.reset_toggled_state()
                    except Exception:
                        pass
                    try:
                        animator.reset_toggle_activation()
                    except Exception:
                        pass
                    # Clear per-session stats
                    _clear_solver_stats_if_any()
                    try:
                        animator.reset_manual()
                    except Exception:
                        pass
                    anim_auto_mode = bool(animator.has_auto())
                    runAnim = False

                else:
                    # Fallback: full rebuild if baseline missing
                    ps = ParticleSystem(config, GGUI=True)
                    solver = ps.build_solver()
                    solver.initialize()
                    animator = AnimationSystem(ps, config)
                    try:
                        solver.time = 0.0
                    except Exception:
                        pass
                    frame_cnt = 0
                    cnt_ply = 0
                    runSim = False
                    anim_time = 0.0
                    # Ensure any toggled bodies start static after full rebuild
                    try:
                        ps.reset_toggled_state()
                    except Exception:
                        pass
                    try:
                        animator.reset_toggle_activation()
                    except Exception:
                        pass
                    anim_auto_mode = bool(animator.has_auto())
                    runAnim = False
                    try:
                        animator.reset_manual()
                    except Exception:
                        pass

            # ----- Animation controls -----
            if anim_auto_mode:
                if window.event.key == 'p':
                    if runSim:
                        runAnim = not runAnim
                if window.event.key == 'o':
                    anim_time = 0.0
            else:
                # Manual mode: discrete press will be ignored; we handle continuous below
                pass

            # ----- Toggle dynamic rigid bodies and/or gated animations -----
            if window.event.key == 't':
                try:
                    oid = -1
                    try:
                        n = len(ps.toggled_ids_sorted)
                        idx = int(ps.toggled_index)
                        while idx < n and (ps.toggled_ids_sorted[idx] in ps.toggled_activated):
                            idx += 1
                        if idx < n:
                            oid = int(ps.toggled_ids_sorted[idx])
                    except Exception:
                        oid = -1
                    if oid != -1:
                        # If this object has an animation, enable it; otherwise switch to dynamic
                        try:
                            if animator.has_animation_for(oid):
                                # Start its animation at 0 at the moment of toggle
                                animator.enable_animation_for_with_time(oid, anim_time)
                                # Ensure auto animation is playing; also apply once immediately
                                try:
                                    animator.apply(anim_time)
                                except Exception:
                                    pass
                                try:
                                    # If any auto anims exist and at least one is enabled, turn on runAnim
                                    anim_auto_mode = bool(animator.has_auto())
                                    if anim_auto_mode:
                                        runAnim = True
                                except Exception:
                                    pass
                                ps.toggled_activated.add(oid)
                                ps.toggled_index = int(ps.toggled_index) + 1
                            else:
                                vel = np.zeros(ps.dim, dtype=np.float32)
                                try:
                                    vel = ps.toggled_dynamic_velocity.get(oid, vel)
                                except Exception:
                                    pass
                                ps._activate_object_dynamic_kernel(int(oid), float(vel[0]), float(vel[1]), float(vel[2]))
                                ps.toggled_activated.add(oid)
                                ps.toggled_index = int(ps.toggled_index) + 1
                                try:
                                    ps.initialize_object_particle_num()
                                except Exception:
                                    pass
                                try:
                                    if ps.num_rigid_bodies > 0:
                                        ps.initialize_rigid_mass()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass

        if (export_ply or export_rigid_mesh) and frame_cnt > end_frame:
            runSim = False
            _export_solver_stats_if_any()

        # Apply animation when paused for immediate feedback
        if not runSim and animator.has_animations():
            try:
                now_wall = time.perf_counter()
                if anim_auto_mode and runAnim:
                    # advance time by wall clock and apply auto animations
                    anim_time += float(now_wall - prev_anim_walltime)
                    animator.apply(anim_time)
                else:
                    animator.apply_manual()
                prev_anim_walltime = now_wall
            except Exception:
                pass

        if runSim:

            dt_frame = solver.dt
            dt_sub = dt_frame / solver.num_substep
            try:
                if getattr(solver, "cfl", False) and hasattr(solver, "compute_cfl_dt"):
                    dt_sub = float(solver.compute_cfl_dt(dt_sub))
            except Exception:
                pass
            solver.dt = dt_sub
            for i in range(solver.num_substep):
                # apply animation each substep depending on mode
                if animator.has_animations():
                    try:
                        if anim_auto_mode and runAnim:
                            t_now = anim_time + i * solver.dt
                            animator.apply(t_now)
                        else:
                            animator.apply_manual()
                    except Exception:
                        pass
                # provide current global substep index to solver for per-iteration frame tagging
                try:
                    solver.current_frame = int(frame_cnt * solver.num_substep + i)
                except Exception:
                    pass
                # Manual continuous input handling for smooth motion (manual mode)
                if not anim_auto_mode:
                    # Determine per-frame delta based on dt
                    trans_delta = anim_nudge * solver.dt
                    rot_delta = rot_nudge * solver.dt
                    # Poll held keys
                    if window.is_pressed(ti.ui.LEFT):
                        # oscillate: x/z -> left/right; rotate: y -> left/right
                        # Apply per-axis according to configured animations
                        try:
                            animator.nudge_translate_axis(0, -trans_delta)  # x-
                            animator.nudge_translate_axis(2, -trans_delta)  # z-
                            animator.nudge_rotate_axis(1, -rot_delta)       # y-
                        except Exception:
                            pass
                    if window.is_pressed(ti.ui.RIGHT):
                        try:
                            animator.nudge_translate_axis(0, +trans_delta)  # x+
                            animator.nudge_translate_axis(2, +trans_delta)  # z+
                            animator.nudge_rotate_axis(1, +rot_delta)       # y+
                        except Exception:
                            pass
                    if window.is_pressed(ti.ui.UP):
                        # oscillate: y -> up/down; rotate: x/z -> up/down
                        try:
                            animator.nudge_translate_axis(1, +trans_delta)  # y+
                            animator.nudge_rotate_axis(0, +rot_delta)       # x+
                            animator.nudge_rotate_axis(2, +rot_delta)       # z+
                        except Exception:
                            pass
                    if window.is_pressed(ti.ui.DOWN):
                        try:
                            animator.nudge_translate_axis(1, -trans_delta)  # y-
                            animator.nudge_rotate_axis(0, -rot_delta)       # x-
                            animator.nudge_rotate_axis(2, -rot_delta)       # z-
                        except Exception:
                            pass
                solver.step()

            # advance time only in auto-running mode
            if anim_auto_mode and runAnim:
                anim_time += dt_frame
            else:
                prev_anim_walltime = time.perf_counter()
            solver.dt = dt_frame
            frame_cnt += 1

            # After completing a frame, cache the end-of-frame state
            cache.push(frame_cnt=frame_cnt, anim_time=anim_time)

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
                                    writer.add_vertex_color(object_id_color[:, 0], object_id_color[:, 1], object_id_color[:, 2])

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

        ps.copy_to_vis_buffer(invisible_objects=invisible_objects)
        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius, color=particle_color)
        elif ps.dim == 3:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))

            # solver.compute_divergence()

            v_np = ps.v.to_numpy()
            density_np = ps.density.to_numpy()
            density0_np = ps.density0.to_numpy()
            div_np = ps.divergence.to_numpy()
            material_np = ps.material.to_numpy()
            dynamic_mask = ps.is_dynamic.to_numpy()

            # v_np = solver.div.to_numpy

            v_norm = np.linalg.norm(v_np, axis = 1)
            # v_norm = solver.div.to_numpy

            # Calculate density ratio relative to rest density (ρ/ρ₀)
            # ρ/ρ₀ > 1: higher density (red)
            # ρ/ρ₀ ≈ 1: normal density (green)
            # ρ/ρ₀ < 1: lower density (blue)
            density = density_np - density0_np

            # Normalize values
            norm_v = Normalize(vmin=0.0, vmax=1.5)
            norm_div = Normalize(vmin=0.0, vmax=5.0)
            norm_density = Normalize(vmin=-50.0, vmax=50.0)
            
            # Step 5: Map normalized values to RGB (fluid particles only)
            if viz_mode == 1:
                if heatmap_type == 1:
                    # Velocity heatmap
                    cmap = LinearSegmentedColormap.from_list("heatmap", ["blue", "white"])
                    rgba_array = cmap(norm_v(v_norm))
                elif heatmap_type == 2:
                    # Divergence heatmap
                    cmap = LinearSegmentedColormap.from_list("heatmap", ["blue", "white", "red"])
                    rgba_array = cmap(norm_div(div_np))
                elif heatmap_type == 3:
                    # Density heatmap
                    cmap = LinearSegmentedColormap.from_list("heatmap", ["blue", "white", "red"])
                    rgba_array = cmap(norm_density(density))
            else:
                rgba_array = cmap(norm_density(density))  # Default for non-heatmap mode

            # Create a color array that only applies heat map to fluid particles
            # Initialize with default colors (from color_vis_buffer)
            default_colors = ps.color_vis_buffer.to_numpy()

            if viz_mode == 1:
                # Heatmap mode
                heat_map_colors = default_colors.copy()
                # Apply heat map colors to fluid particles and dynamic rigid bodies
                fluid_mask = (material_np == ps.material_fluid)
                show_mask = np.logical_or(fluid_mask, dynamic_mask)
                heat_map_colors[show_mask] = rgba_array[show_mask]

                # Apply transparency to specific objects
                object_id_np = ps.object_id.to_numpy()
                for obj_id in transparent_objects:
                    obj_mask = (object_id_np == obj_id)
                    heat_map_colors[obj_mask, 3] = color_alpha # Set alpha to 0.2 for transparent objects

                # Filter out particles with very low alpha values for true transparency
                if color_alpha < 0.01:
                    transparent_mask = np.zeros_like(heat_map_colors[:, 0], dtype=bool)
                    for obj_id in transparent_objects:
                        obj_mask = (object_id_np == obj_id)
                        transparent_mask |= obj_mask

                    # Set positions of transparent particles to far away so they're not rendered
                    transparent_positions = ps.x.to_numpy()
                    transparent_positions[transparent_mask] = [1000.0, 1000.0, 1000.0]  # Move far away
                    ps.x_vis_buffer.from_numpy(transparent_positions)

                    # Also set alpha to 0 for shadow casting purposes
                    heat_map_colors[transparent_mask, 3] = 0.0
                else:
                    # For semi-transparent objects, reduce shadow intensity based on alpha
                    for obj_id in transparent_objects:
                        obj_mask = (object_id_np == obj_id)
                        # Scale alpha for shadow casting - lower alpha means less shadow
                        shadow_alpha = color_alpha * 0.5  # Reduce shadow intensity
                        heat_map_colors[obj_mask, 3] = shadow_alpha

                ps.color_heat_map.from_numpy(heat_map_colors)
                render_colors = ps.color_heat_map

            else:
                # Original colors mode
                original_colors = default_colors.copy()

                # Apply transparency to specific objects
                object_id_np = ps.object_id.to_numpy()
                for obj_id in transparent_objects:
                    obj_mask = (object_id_np == obj_id)
                    original_colors[obj_mask, 3] = color_alpha

                # Filter out particles with very low alpha values for true transparency
                if color_alpha < 0.01:
                    transparent_mask = np.zeros_like(original_colors[:, 0], dtype=bool)
                    for obj_id in transparent_objects:
                        obj_mask = (object_id_np == obj_id)
                        transparent_mask |= obj_mask

                    # Set positions of transparent particles to far away so they're not rendered
                    transparent_positions = ps.x.to_numpy()
                    transparent_positions[transparent_mask] = [1000.0, 1000.0, 1000.0]  # Move far away
                    ps.x_vis_buffer.from_numpy(transparent_positions)

                    # Also set alpha to 0 for shadow casting purposes
                    original_colors[transparent_mask, 3] = 0.0
                else:
                    # For semi-transparent objects, reduce shadow intensity based on alpha
                    for obj_id in transparent_objects:
                        obj_mask = (object_id_np == obj_id)
                        # Scale alpha for shadow casting - lower alpha means less shadow
                        original_colors[obj_mask, 3] = shadow_alpha

                ps.color_heat_map.from_numpy(original_colors)
                render_colors = ps.color_heat_map

            # print(rgb_array.dtype)
            # print(rgb_array)
            # ps.color_heat_map.from_numpy(heat_map_colors)
            # print(rgb_array.shape)

            scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=render_colors)
            # scene.particles(ps.x, radius=ps.particle_radius, per_vertex_color=ps.color_heat_map)
            # scene.particles(ps.x, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)

            scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28, 1.0), width = 1.0)
            canvas.scene(scene)
    
        if output_frames:
            if cnt % output_interval == 0:
                window.write_image(f"{scene_name}_output_img/{cnt:06}.png")

        cnt += 1
        # if cnt > 6000:
        #     break
        window.show() 

