import tqdm
from utils.keyboard_controller import KeyBoardController
from utils.vision import get_fly_vision, get_fly_vision_raw, render_image_with_vision
from flygym import Fly, YawOnlyCamera, SingleFlySimulation
from flygym.arena import FlatTerrain
from cobar_miniproject import levels

import utils.cobar_fly as cobar_fly
import cv2
import sys

import cobar_miniproject

# OPTIONS
#
# render the camera
# render the fly vision
# render the raw fly vision
#

if __name__ == "__main__":
    if len(sys.argv) > 1:
        level = int(sys.argv[1])
    else:
        level = -1

    timestep = 1e-4

    # you can pass in parameters to enable different senses here
    # head stabilisation
    # camera could be optional - just play in fly mode
    fly = cobar_fly.CobarFly(debug=True, enable_vision=True)

    if level <= -1:
        level_arena = FlatTerrain()
    elif level <= 1:
        level_arena = levels[level]()
    else:
        level_arena = levels[level](timestep=timestep, fly=fly)

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

    controller = KeyBoardController(timestep=timestep, seed=0)

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
            progress_bar.update()

    print("Simulation finished")

    # Save video
    cam.save_video("./outputs/hybrid_controller.mp4", 0)
