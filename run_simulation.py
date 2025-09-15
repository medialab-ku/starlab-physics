import os
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from particle_system import ParticleSystem
from animation import AnimationSystem
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

    substeps = config.get_cfg("numSubstepping")
    # print(substeps)
    output_frames = config.get_cfg("exportFrame")
    output_interval = int(0.02 / config.get_cfg("timeStepSize"))
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
    if output_frames:
        os.makedirs(f"{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"{scene_name}_output", exist_ok=True)

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
    end_frame = 1000

    # Initialize animation system
    animator = AnimationSystem(ps, config)
    anim_time = 0.0

    # Animation handled by AnimationSystem

    def show_options_solver():

        with gui.sub_window("Solver settings", 0., 0., 0.4, 0.3) as w:

            solver.dt = w.slider_float("dt", solver.dt, 0.001, 0.04)
            solver.cfl = w.checkbox("CFL", solver.cfl)
            solver.num_substep = w.slider_int("substepping", solver.num_substep, 1, 100)

            if method == 2:
                solver.tol_opt = w.slider_int("opt tol magnitude", solver.tol_opt, 1, 5)
                solver.max_iteration_opt = w.slider_int("max opt. iter", solver.max_iteration_opt, 1, 1000)

                # solver.enable_DF = w.checkbox("divergence-free solve", solver.enable_DF)
                solver.smooth_max = w.checkbox("smooth max(Ours)", solver.smooth_max)
                if solver.smooth_max:
                    solver.eps = w.slider_float("eps", solver.eps, 0.001, 10.0)
                    solver.use_pcg = w.checkbox("PCG", solver.use_pcg)

                    if solver.use_pcg:
                        solver.max_iteration_pcg = w.slider_int("max pcg. iter", solver.max_iteration_pcg, 1, 1000)
                        solver.tol_pcg = w.slider_int("pcg tol magnitude", solver.tol_pcg, 1, 5)

                else:
                    solver.omega = w.slider_float("relaxation", solver.omega, 0.001, 2.0)

            gui.text(f"# fluid particle: {ps.fluid_particle_num}")
            gui.text(f"# boundary particle: {ps.solid_particle_num}")
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

        with gui.sub_window("Visualization settings", 0.0, 0.4, 0.4, 0.3) as w:

            export_ply = w.checkbox("export", export_ply)
            if export_ply:
                export_rigid_objects = w.checkbox("Export rigid objects", export_rigid_objects)

            if export_ply:
                end_frame = w.slider_int("end frame", end_frame, 0, int(5e3))

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


    def show_options_stats():

        with gui.sub_window("Stats. settings", 0.7, 0.0, 0.3, 0.2) as w:

            solver.print_opt_iter  = w.checkbox("print opt iter", solver.print_opt_iter)
            solver.print_opt_error = w.checkbox("print opt error", solver.print_opt_error)
            solver.print_pcg_iter  = w.checkbox("print pcg iter", solver.print_pcg_iter)
            solver.print_pcg_error = w.checkbox("print pcg error", solver.print_pcg_error)
            solver.print_elapsed_time = w.checkbox("print elapsed time", solver.print_elapsed_time)

    cnt = 0
    cnt_ply = 0
    runSim = False

    @ti.kernel
    def reset_R_identity(R: ti.template()):
        for i in ti.grouped(R):
            R[i] = ti.math.mat3([[1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])

    while window.running:

        show_options_solver()
        show_options_visual()
        show_options_stats()

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                runSim = not runSim

            if window.event.key == 'r':
                print("rest simulation...")
                # Preserve solver settings that can be tuned via GUI
                preserve_names = [
                    'dt', 'num_substep', 'cfl',
                    'tol_opt', 'max_iteration_opt', 'method', 'print_info', 'print_opt_iter', 'print_opt_error', 'print_pcg_iter', 'print_pcg_error', 'print_elapsed_time',
                    'omega', 'iisph_vanilla', 'smooth_max', 'eps', 'use_pcg',
                    'max_iteration_pcg', 'tol_pcg', 'adaptive_step_size',
                ]
                preserved = {}
                for name in preserve_names:
                    if hasattr(solver, name):
                        try:
                            preserved[name] = getattr(solver, name)
                        except Exception:
                            pass

                # Rebuild particle system and solver to fully drop emitted particles/state
                ps = ParticleSystem(config, GGUI=True)
                solver = ps.build_solver()
                solver.initialize()
                animator = AnimationSystem(ps, config)

                # Restore preserved settings
                for name, value in preserved.items():
                    try:
                        if hasattr(solver, name):
                            setattr(solver, name, value)
                    except Exception:
                        pass
                try:
                    solver.time = 0.0
                except Exception:
                    pass

                # Reset counters and pause sim
                frame_cnt = 0
                cnt_ply = 0
                runSim = False
                anim_time = 0.0

        if export_ply and frame_cnt > end_frame:
            runSim = False

        if runSim:

            # Move boundary object if the scene is moving_boundary
            dt = solver.dt
            solver.dt = dt / solver.num_substep
            for i in range(solver.num_substep):
                # advance animation within substeps for smoother motion
                if animator.has_animations():
                    t_now = anim_time + i * solver.dt
                    animator.apply(t_now)
                solver.step()

            # advance animation time by full frame dt
            if animator.has_animations():
                anim_time += dt
            solver.dt = dt
            frame_cnt += 1

            if frame_cnt > 0 and frame_cnt % output_interval == 0:
                if export_ply:
                    if export_rigid_objects:
                        # Export each object separately
                        for obj_id in ps.object_collection:
                            obj_data = ps.dump(obj_id=obj_id)
                            np_pos = obj_data["position"]

                            # Only export if object has particles
                            if len(np_pos) > 0:
                                if obj_id == 0:
                                    # Fluid particles (object id 0): position + RGB encoding velocity x,y,z
                                    np_vel = obj_data["velocity"]

                                    # Normalize velocity components to 0-1 range for RGB encoding
                                    # Assume velocity range is roughly -5 to 5
                                    vel_x_norm = np.clip((np_vel[:, 0] + 5.0) / 10.0, 0.0, 1.0)
                                    vel_y_norm = np.clip((np_vel[:, 1] + 5.0) / 10.0, 0.0, 1.0)
                                    vel_z_norm = np.clip((np_vel[:, 2] + 5.0) / 10.0, 0.0, 1.0)

                                    # Create separate PLY file for fluid
                                    obj_series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, obj_id)
                                    writer = ti.tools.PLYWriter(num_vertices=len(np_pos))
                                    writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])

                                    # Encode velocity x,y,z in RGB channels
                                    writer.add_vertex_color(vel_x_norm, vel_y_norm, vel_z_norm)

                                    writer.export_frame_ascii(cnt_ply, obj_series_prefix)
                                else:
                                    # Rigid objects (object id > 0): position only
                                    obj_series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, obj_id)
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

                            # Normalize velocity components to 0-1 range for RGB encoding
                            # Assume velocity range is roughly -5 to 5
                            vel_x_norm = np.clip((np_vel[:, 0] + 5.0) / 10.0, 0.0, 1.0)
                            vel_y_norm = np.clip((np_vel[:, 1] + 5.0) / 10.0, 0.0, 1.0)
                            vel_z_norm = np.clip((np_vel[:, 2] + 5.0) / 10.0, 0.0, 1.0)

                            writer = ti.tools.PLYWriter(num_vertices=len(np_pos))
                            writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])

                            # Encode velocity x,y,z in RGB channels
                            writer.add_vertex_color(vel_x_norm, vel_y_norm, vel_z_norm)

                            writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
                    cnt_ply += 1

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

                # Apply heat map colors only to fluid particles
                fluid_mask = (material_np == ps.material_fluid)
                heat_map_colors[fluid_mask] = rgba_array[fluid_mask]

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
        
        if output_obj:
            for r_body_id in ps.object_id_rigid_body:
                with open(f"{scene_name}_output/obj_{r_body_id}_{cnt_ply:06}.obj", "w") as f:
                    e = ps.object_collection[r_body_id]["mesh"].export(file_type='obj')
                    f.write(e)

        cnt += 1
        # if cnt > 6000:
        #     break
        window.show()

