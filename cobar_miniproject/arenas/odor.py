import numpy as np
from flygym.arena import FlatTerrain
from flygym.examples.vision.arena import ObstacleOdorArena
from .utils import get_random_pos, circ


class OdorArena(ObstacleOdorArena):
    def __init__(
        self,
        timestep,
        fly,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        seed=None,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)

        target_position = get_random_pos(
            distance_range=target_distance_range,
            angle_range=target_angle_range,
            random_state=rng,
        )

        super().__init__(
            terrain=FlatTerrain(ground_alpha=0),
            obstacle_positions=np.array([]),
            peak_odor_intensity=np.array([[1, 0]]),
            odor_source=np.array([[*target_position, 1]]),
            marker_colors=np.array([target_marker_color]),
            marker_size=target_marker_size,
            **kwargs,
        )
