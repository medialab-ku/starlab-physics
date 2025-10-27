import argparse
import taichi as ti
from config_builder import SimConfig
from scene_loader import SceneLoader
from simulation_data import SimulationData
from framework import Framework
from neighbour_search import NeighborSearch
from pressure import Pressure
from surface_tension import SurfaceTension
from viscosity import Viscosity
from elasticity import Elasticity
from visualization import VisualizationEngine, VisualizationSettings, ColorMode, HeatmapField
from output_manager import OutputManager, OutputConfig, ExportFormat
from cache_system import SimulationCache
from randomizer import ParticleRandomizer
from pinning import ParticlePinning
from animation import AnimationEngine

ti.init(arch=ti.gpu, device_memory_fraction=0.7)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)

    substeps = config.get_cfg("numSubstepping")
    loader = SceneLoader(config)
    scene_name = loader.get_scene_name(scene_path)
    scene_data = loader.prepare_scene()


    ps = SimulationData(config, GGUI=True)
    loader.populate_scene(ps, scene_data)
    loader.reset_emitter_system()

    neighbor_search = NeighborSearch(config, ps)
    pressure = Pressure(ps)
    viscosity = Viscosity(ps)
    surface_tension = SurfaceTension(ps)
    elasticity = Elasticity(ps)
    pin_util = ParticlePinning(ps)
    pin_geom = pin_util.apply(scene_data) # guide lines

    anim = AnimationEngine(ps)
    anim.build(scene_data, pin_geom)

    fw = Framework(ps, neighbor_search, pressure, viscosity, surface_tension, elasticity)
    fw.initialize()

    anim_time = 0.0

    randomizer = ParticleRandomizer(ps, neighbor_search)

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
    is_invisible = False

    viz_settings = VisualizationSettings(
        color_mode=ColorMode.heatmap,
        heatmap_field={1: HeatmapField.velocity, 2: HeatmapField.divergence, 3: HeatmapField.density}[3],
        transparent_objects=transparent_objects,
        invisible_objects=invisible_objects,
        color_alpha=color_alpha,
    )
    viz = VisualizationEngine(ps, config, viz_settings)

    output_cfg = OutputConfig(
        export_particles=False,
        selected_format=ExportFormat.ply,
        export_fluid_particles=True,
        export_rigid_particles=False,
        export_mesh_obj=False,
        frame_interval=20,
        include_heatmap_attributes=True,
        end_frame=600,
    )
    output_manager = OutputManager(scene_name, output_cfg)
    
    frame_cnt = 0
    # export_ply = output_ply

    if pin_geom and pin_geom.get("vertices") is not None:
        viz.set_pin_lines(pin_geom["vertices"], pin_geom["indices"])

    # Caching system
    cache = SimulationCache(ps, fw, max_steps=10)
    cache.snapshot_baseline(anim_time=anim_time)


    def show_options_solver():
        with gui.sub_window("Solver settings", 0., 0., 0.4, 0.4) as w:
            fw.dt = w.slider_float("dt", fw.dt, 0.001, 0.04)
            fw.alpha = w.slider_float("alpha", fw.alpha, 0.0, 10.0)
            fw.YM = w.slider_float("YM", fw.YM, 1e6, 10e6)
            fw.PR = w.slider_float("PR", fw.PR, 0.0, 0.499)
            try:
                N_active = int(ps.particle_num[None])
                mats = ps.material.to_numpy()[:N_active]
                fluid_cnt = int((mats == ps.material_fluid).sum())
                deform_cnt = int((mats == ps.material_solid).sum())
            except Exception:
                fluid_cnt = int(ps.fluid_particle_num)
                deform_cnt = int(ps.solid_particle_num) 
            gui.text(f"# fluid particle: {fluid_cnt}")
            gui.text(f"# deformable particle: {deform_cnt}") 
            gui.text(f"# boundary particle: {ps.rigid_particle_num}")
            gui.text(f"Current frame: {frame_cnt}")


    def show_options_visual():
        with gui.sub_window("Visualization settings", 0.0, 0.4, 0.4, 0.3) as w:
            gui.text("")  # Spacer
            gui.text("Visualization Controls:")
            viz.render_ui(w, gui)


    def show_options_export():
        with gui.sub_window("Export settings", 0.0, 0.7, 0.4, 0.2) as w:
            gui.text("")
            gui.text("Export settings:")
            output_manager.render_ui(w, gui, ps)
    def show_options_cache():
        return cache.show_ui(gui, current_frame=frame_cnt, pos=(0.0, 0.9), size=(0.4, 0.25))

    cnt = 0
    cnt_ply = 0
    cnt_obj = 0
    runSim = False
    expand_mode = False
    expand_factor = 4.0
    rand_step_alpha = 0.01
    random_seed = 1337

    while window.running:

        show_options_solver()
        show_options_visual()
        show_options_export()
        # show_options_stats()
        # Cache UI
        res_cache = show_options_cache()
        if res_cache.get("restored", False):
            runSim = False
            frame_cnt = int(res_cache.get("frame", frame_cnt))

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                if expand_mode:
                    expand_mode = False
                # Toggle run state
                runSim = not runSim
                if runSim:
                    ps.x_old.copy_from(ps.x)
                    ps.v_adv.copy_from(ps.v)
                    viz.update_buffers()
                    output_manager.on_step(frame_cnt, ps, viz)
                    cache.snapshot_baseline(anim_time=anim_time)

            if window.event.key == 'b':
                # Rewind one cached frame
                runSim = False
                result = cache.rewind_one(frame_cnt)
                if result.get("restored", False):
                    frame_cnt = int(result.get("frame", frame_cnt))
                    anim_time = float(result.get("anim_time", anim_time) or 0.0)
                    fw.initialize()
                    print(f"rewind: {result.get('rewind_steps', 0)} frames")

            if window.event.key == 'p':
                expand_mode = True
                randomizer.begin(expand=expand_factor, seed=random_seed)
                runSim = True
                print("Randomize...")

            if window.event.key == 'r':
                print("reset simulation...")
                ok, _, _ = (False, None, None)
                ok, frame_restored, anim_time_restored = cache.restore_baseline()

                if ok:
                    # Reset GUI counters and pause
                    frame_cnt = frame_restored
                    anim_time = anim_time_restored
                    cnt_ply = 0
                    runSim = False
                    fw.initialize()

        output_cfg = output_manager.get_config()
        if (output_cfg.export_particles or output_cfg.export_mesh_obj) and frame_cnt > int(output_cfg.end_frame):
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

            if expand_mode:
                randomizer.step(alpha=rand_step_alpha)
                if not randomizer.active:
                    expand_mode = False
                    runSim = False
                    print("Randomize completed. Press SPACE to resume physics.")
            else:
                fw.forward()
                anim.apply(anim_time, dt_sub)
                anim_time += dt_sub

            fw.dt = dt_frame

            if not expand_mode:
                frame_cnt += 1
                cache.push(frame_cnt=frame_cnt, anim_time=anim_time)

        viz.update_buffers()
        output_manager.on_step(frame_cnt, ps, viz)

        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius, color=particle_color)
        else:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
            scene.set_camera(camera)
            scene.point_light((2.5, 6.0, 2.5), color=(1.0, 1.0, 1.0))
            viz.draw(scene, canvas, background_color=background_color)

        cnt += 1
        # if cnt > 6000:
        #     break
        window.show()
