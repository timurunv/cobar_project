import argparse
import cv2
import tqdm
from cobar_miniproject.arenas import level_arenas
from cobar_miniproject.keyboard_controller import KeyBoardController
from cobar_miniproject.cobar_fly import CobarFly
from cobar_miniproject.vision import (
    get_fly_vision,
    get_fly_vision_raw,
    render_image_with_vision,
)
from flygym import YawOnlyCamera, SingleFlySimulation
from flygym.arena import FlatTerrain

# OPTIONS
#
# render the camera
# render the fly vision
# render the raw fly vision
#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the fly simulation.")
    parser.add_argument(
        "--level", type=int, default=-1, help="Level to load (default: -1)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the simulation (default: 0)"
    )
    args = parser.parse_args()

    level = args.level
    seed = args.seed
    timestep = 1e-4

    # you can pass in parameters to enable different senses here
    # head stabilisation
    # camera could be optional - just play in fly mode
    fly = CobarFly(debug=True, enable_vision=False)

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        level_arena = level_arenas[level](fly=fly)
    else:
        level_arena = FlatTerrain()

    cam = YawOnlyCamera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_back_track_game",
        targeted_fly_names=[fly.name],
        play_speed=0.2,
    )

    sim = SingleFlySimulation(
        fly=fly,
        cameras=[cam],
        timestep=timestep,
        arena=level_arena,
    )

    controller = KeyBoardController(timestep=timestep, seed=seed)

    # run cpg simulation
    obs, info = sim.reset()
    obs_hist = []
    info_hist = []

    # create window
    cv2.namedWindow("Simulation", cv2.WINDOW_NORMAL)

    with tqdm.tqdm(desc="running simulation") as progress_bar:
        while True:
            # Get observations
            obs, reward, terminated, truncated, info = sim.step(
                controller.get_actions(obs)
            )
            if controller.done_level(obs):
                # finish the path integration level
                break

            obs_hist.append(obs)
            info_hist.append(info)

            rendered_img = sim.render()[0]
            if rendered_img is not None:
                rendered_img = render_image_with_vision(
                    rendered_img, get_fly_vision(fly)
                )
                rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
                cv2.imshow("Simulation", rendered_img)
                cv2.waitKey(1)

            if controller.quit:
                print("Simulation terminated by user.")
                break
            if level_arena.quit:
                print("Target reached. Simulation terminated.")
                break

            progress_bar.update()

    print("Simulation finished")

    # Save video
    cam.save_video("./outputs/hybrid_controller.mp4", 0)
