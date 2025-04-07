import cv2
import numpy as np
from flygym.arena import FlatTerrain
from flygym.examples.vision.arena import ObstacleOdorArena

def get_random_target_position(r_range, theta_range, rng):
    p = rng.uniform(*r_range) * np.exp(1j * rng.uniform(*theta_range))
    return np.array([p.real, p.imag])

def circ(im, xy, r, value, xmin, ymin, res, outer=False):
    c = ((np.asarray(xy) - (xmin, ymin)) / res).astype(int)
    r = int(r / res)
    if outer:
        cv2.circle(im, center=c, radius=r + 1, color=value, thickness=1)
    else:
        cv2.circle(im, center=c, radius=r, color=value, thickness=-1)

class ScatteredPillarsArena(ObstacleOdorArena):
    def __init__(
        self,
        r_range=(29, 31),
        theta_range=(-np.pi, np.pi),
        pillar_height=3,
        pillar_radius=0.3,
        pillar_separation=6,
        target_space=4,
        fly_space=4,
        target_size=0.3,
        target_color=(1, 0.5, 14 / 255, 1),
        seed=None,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)

        target_position = get_random_target_position(
            r_range=r_range,
            theta_range=theta_range,
            rng=rng,
        )

        pillar_positions = self.get_pillar_positions(
            rng=rng,
            target_position=target_position,
            pillar_radius=pillar_radius,
            target_space=target_space,
            pillar_separation=pillar_separation,
            fly_space=fly_space,
        )

        super().__init__(
            odor_source=np.array([[*target_position, 1]]),
            marker_colors=np.array([target_color]),
            peak_odor_intensity=np.array([[1, 0]]),
            obstacle_positions=pillar_positions,
            obstacle_radius=pillar_radius,
            obstacle_height=pillar_height,
            terrain=FlatTerrain(ground_alpha=0),
            marker_size=target_size,
            **kwargs,
        )

    @staticmethod
    def get_pillar_positions(
        rng,
        target_position,
        pillar_radius,
        target_space,
        pillar_separation,
        fly_space,
    ):
        pillar_space = pillar_radius * 2 + pillar_separation
        vmax = np.linalg.norm(target_position)
        xmin = ymin = -vmax
        xmax = ymax = vmax
        res = 0.05  # resolution of the grid
        n_cols = int((xmax - xmin) / res)
        n_rows = int((ymax - ymin) / res)
        im = np.zeros((n_rows, n_cols), dtype=np.uint8)
        circ(im, (0, 0), vmax, 255, xmin, ymin, res)
        circ(im, (0, 0), fly_space, 0, xmin, ymin, res)
        circ(im, target_position, target_space, 0, xmin, ymin, res)
        pillar0_position = target_position / 2
        circ(im, pillar0_position, pillar_space, 0, xmin, ymin, res)
        im2 = np.zeros((n_rows, n_cols), dtype=np.uint8)
        circ(im2, pillar0_position, pillar_space, 255, xmin, ymin, res, outer=True)
        pillar_positions = [pillar0_position]

        while True:
            argwhere = np.argwhere(im & im2)
            try:
                p = argwhere[rng.choice(len(argwhere)), ::-1] * res + (xmin, ymin)
            except ValueError:
                break
            pillar_positions.append(p)
            circ(im, p, pillar_space, 0, xmin, ymin, res)
            circ(im2, p, pillar_space, 255, xmin, ymin, res, outer=True)
        
        return np.array(pillar_positions)
