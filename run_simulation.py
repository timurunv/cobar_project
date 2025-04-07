import sys
from pathlib import Path
from flygym import Fly, Camera, SingleFlySimulation
from cobar_miniproject import levels
import importlib


def run_simulation(
    submission_dir,
    level,
    seed,
    max_steps,
):
    from tqdm import trange

    submission_dir = Path(submission_dir)
    sys.path.append(str(submission_dir.parent))
    module = importlib.import_module(submission_dir.name)
    controller = module.controller.Controller()

    fly = Fly(
        render_raw_vision=True,
        enable_vision=True,
        enable_adhesion=True,
        xml_variant="seqik_simple",
    )

    arena = levels[level](seed=seed)

    cam_params = {
        "class": "nmf",
        "mode": "track",
        "ipd": 0.068,
        "pos": [0, 0, 80],
        "euler": [0, 0, 0],
    }

    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name="camera_top_zoomout",
        camera_parameters=cam_params,
    )

    sim = SingleFlySimulation(
        fly=fly,
        arena=arena,
        cameras=[cam],
    )

    obs, info = sim.reset(seed=0)

    for _ in trange(max_steps):
        obs["raw_vision"] = info["raw_vision"]
        sim.step(controller(obs))
        sim.render()

    output_dir = Path("outputs")
    video_path = output_dir / f"{submission_dir.name}_{level}_{seed}.mp4"
    cam.save_video(video_path, stabilization_time=0)


if __name__ == "__main__":
    fixed_seeds = [0, 42, 1337]  # these seeds are fixed
    secret_seeds = []  # there will be 7 more secret seeds
    seeds = fixed_seeds + secret_seeds
    submission_directories = [
        i for i in Path("submissions").glob("group*") if i.is_dir()
    ]
    max_steps = 10000

    for submission_dir in submission_directories:
        for level in range(1):
            for seed in seeds:
                print(
                    f"Running simulation for {submission_dir.name}, level {level}, seed {seed}"
                )
                run_simulation(submission_dir, level, seed, max_steps)
