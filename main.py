from utils.keyboard_controller import KeyBoardController
from utils.vision import get_fly_vision, get_fly_vision_raw, render_image_with_vision
from flygym import Fly, YawOnlyCamera, SingleFlySimulation
from flygym.arena import FlatTerrain
import utils.cobar_fly as cobar_fly
import cv2
from tqdm import trange
import numpy as np

if __name__ == "__main__":
    run_time = 5.0
    timestep = 1e-4

    # Initialize the simulation
    # fly = Fly(
    #     enable_adhesion=True,
    #     draw_adhesion=True,
    #     init_pose="stretch",
    #     control="position",
    #     xml_variant="seqik_simple",
    # )

    fly = cobar_fly.SimpleHeadStabilisedFly(enable_vision=True)

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
        arena=FlatTerrain(),
    )

    controller = KeyBoardController(timestep=timestep, seed=0)

    # run cpg simulation
    obs, info = sim.reset()
    print(f"Spawning fly at {obs['fly'][0]} mm")
    target_num_steps = int(run_time / sim.timestep)
    obs_hist = []
    info_hist = []

    #create window
    cv2.namedWindow("Simulation", cv2.WINDOW_NORMAL)

    for k in trange(target_num_steps):
        # Get observations
        obs, reward, terminated, truncated, info = sim.step(controller.get_actions(obs))
        obs_hist.append(obs)
        info_hist.append(info)

        rendered_img = sim.render()[0]
        if rendered_img is not None:
            rendered_img = render_image_with_vision(rendered_img, get_fly_vision(fly))
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB)
            cv2.imshow("Simulation", rendered_img)
            cv2.waitKey(1)

        if controller.quit:
            print("Simulation terminated by user.")
            break


    print(f"Simulation terminated: {obs_hist[-1]['fly'][0] - obs_hist[0]['fly'][0]}")

    # Save video
    cam.save_video(f"./outputs/hybrid_controller.mp4", 0)
