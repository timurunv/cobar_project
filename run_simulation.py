from pathlib import Path
import importlib
import argparse
import sys
from tqdm import trange
from flygym import Camera
from cobar_miniproject import levels
from cobar_miniproject.cobar_fly import CobarFly
from cobar_miniproject.vision import (
    get_fly_vision,
    get_fly_vision_raw,
    render_image_with_vision,
)
from flygym import Camera, SingleFlySimulation
from flygym.arena import FlatTerrain
from tqdm import tqdm
from submission.utils import plot_trajectory
from matplotlib import pyplot as plt
import cv2


def run_simulation(
    submission_dir,
    level,
    seed,
    debug,
    max_steps,
    output_dir="outputs",
    progress=True,
    save_video=True,
    save_plot=False,
):
    sys.path.append(str(submission_dir.parent))
    module = importlib.import_module(submission_dir.name)
    controller = module.controller.Controller()
    timestep = 1e-4

    fly = CobarFly(
        debug=debug,
        enable_vision=True,
        render_raw_vision=True, # can be changed to False for faster simulation
    )

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        # levels 0 and 1 don't need the timestep
        level_arena = levels[level](fly=fly, seed=seed)
    else:
        # levels 2-4 need the timestep
        level_arena = levels[level](fly=fly, timestep=timestep, seed=seed)
    
    cam_params = {"pos": (0, 0, 80)}

    cam = Camera(
        attachment_point=level_arena.root_element.worldbody,
        camera_name="camera_top_zoomout",
        targeted_fly_names=[fly.name],
        camera_parameters=cam_params,
        play_speed=1.0,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    # run cpg simulation
    obs, info = sim.reset()
    obs_hist = []
    info_hist = []

    if progress:
        step_range = trange(max_steps)
    else:
        step_range = range(max_steps)

    counter = 0
    for i in step_range:
        # Get observations
        obs, reward, terminated, truncated, info = sim.step(controller.get_actions(obs))
        rendered_img = sim.render()[0]
        RENDER_TYPE = "RAW_VISION"
        if False: # rendered_img is not None:
            if RENDER_TYPE == "RAW_VISION":
                rendered_img = render_image_with_vision(
                    rendered_img, get_fly_vision(fly), obs["odor_intensity"],
                )
            else:
                rendered_img = render_image_with_vision(
                    rendered_img, get_fly_vision_raw(fly), obs["odor_intensity"],
                )
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
            cv2.imshow("Simulation", rendered_img)
            cv2.waitKey(1)
        if controller.done_level(obs):
            # finish the path integration level
            break

        obs_ = obs.copy()
        if not obs_["vision_updated"]: # to save memory
            if "vision" in obs_:
                del obs_["vision"]
        if "raw_vision" in obs_:
            del obs_["raw_vision"]
        obs_hist.append(obs_)
        #info_hist.append(info)
        if obs.get("reached_odour", False):
            counter += 1
        
        if counter == 500:
            controller.quit =  True

        if hasattr(controller, "quit") and controller.quit:
            print("Simulation terminated by user.")
            break
        if hasattr(level_arena, "quit") and level_arena.quit:
            print("Target reached. Simulation terminated.")
            break
        if i > 2000: 
            obs['reached_odour'] = True
        
        

    if save_video: # Save video
        save_path = Path(output_dir) / f"level{level}_seed{seed}_iter{max_steps}.mp4"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cam.save_video(save_path, stabilization_time=0)
    if save_plot: # Save a plot of the trajectory
        save_path = Path(output_dir) / f"level{level}_seed{seed}_iter{max_steps}.png"
        plot_trajectory(save_path, obs_hist, level_arena.obstacle_positions, level_arena.odor_source, level_arena.obstacle_radius, level_arena.odor_dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "--submission_dir",
        type=Path,
        help="Path to the submission directory containing the controller module.",
        default=str("./submission/"),
    )
    parser.add_argument(
        "--level",
        type=int,
        help="Simulation level to run (e.g., -1 for FlatTerrain, 0-4 for specific levels).",
        default=0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for the simulation.",
        default=19,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of steps to run the simulation.",
        default=10000,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for the simulation.",
        default=False,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the simulation outputs (default: 'outputs').",
        default="outputs",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show progress bar during simulation.",
        default=True,
    )
    parser.add_argument(
        "--savevid",
        action="store_true",
        help="Save the video at the end of the simulation.",
    )
    parser.add_argument(
        "--saveplot",
        action="store_true",
        help="Save a trajectory plot at the end of the simulation.",
    )
    args = parser.parse_args()
    if args.saveplot:
        args.debug = True
    run_simulation(
        submission_dir=args.submission_dir,
        level=args.level,
        seed=args.seed,
        debug=args.debug,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        progress=args.progress,
        save_video=args.savevid,
        save_plot=args.saveplot,
    )
