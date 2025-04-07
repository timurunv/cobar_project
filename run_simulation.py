from tqdm import tqdm
import sys
from pathlib import Path
from flygym import Fly, Camera, SingleFlySimulation
from cobar_miniproject import levels
from tqdm import trange
import importlib


def run_simulation(
    submission_dir,
    level,
    seed,
    max_steps,
):
    submission_dir = Path(submission_dir)
    print(
        f"Running simulation for {submission_dir.name} at level {level} with seed {seed}"
    )
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

    cam_params = {"pos": (0, 0, 80)}

    cam = Camera(
        attachment_point=arena.root_element.worldbody,
        targeted_fly_names=[fly.name],
        camera_name="camera_top_zoomout",
        camera_parameters=cam_params,
        window_size=(640, 640),
        play_speed=0.1,
        fps=30,
    )

    sim = SingleFlySimulation(
        fly=fly,
        arena=arena,
        cameras=[cam],
    )

    obs, info = sim.reset(seed=0)

    output_dir = Path("outputs")
    video_path = output_dir / f"{submission_dir.name}_{level}_{seed}.mp4"
    video_path.parent.mkdir(exist_ok=True, parents=True)

    for _ in trange(max_steps):
        obs["raw_vision"] = info["raw_vision"]
        sim.step(controller(obs))
        sim.render()[0]

    cam.save_video(video_path, stabilization_time=0)


if __name__ == "__main__":
    from joblib import Parallel, delayed
    from itertools import product

    fixed_seeds = [0, 42, 1337]  # these seeds are fixed
    secret_seeds = []  # there will be 7 more secret seeds
    seeds = fixed_seeds + secret_seeds
    root_dir = Path(__file__).parent / "submissions"
    submission_directories = [i for i in root_dir.glob("group*") if i.is_dir()]
    max_steps = 10000

    arg_list = list(
        product(
            submission_directories,
            range(2),
            seeds,
            [max_steps],
        )
    )

    Parallel(n_jobs=-2)(delayed(run_simulation)(*args) for args in tqdm(arg_list))
