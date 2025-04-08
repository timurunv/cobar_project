import cv2
import numpy as np
from flygym.arena import FlatTerrain
from flygym.examples.vision.arena import ObstacleOdorArena


def _get_random_target_position(
    distance_range: tuple[float, float],
    angle_range: tuple[float, float],
    rng: np.random.Generator,
):
    """Generate a random target position.

    Parameters
    ----------
    distance_range : tuple[float, float]
        Distance range from the origin.
    angle_range : tuple[float, float]
        Angle rabge in radians.
    rng : np.random.Generator
        The random number generator.
    Returns
    -------
    np.ndarray
        The target position in the form of [x, y].
    """
    p = rng.uniform(*distance_range) * np.exp(1j * rng.uniform(*angle_range))
    return np.array([p.real, p.imag], float)


def _circ(
    img: np.ndarray,
    xy: tuple[float, float],
    r: float,
    value: bool,
    xmin: float,
    ymin: float,
    res: float,
    outer=False,
):
    """Draw a circle on a 2D image.

    Parameters
    ----------
    img : np.ndarray
        The image to draw on.
    xy : tuple[float, float]
        The center of the circle.
    r : float
        The radius of the circle.
    value : bool
        The value to set the pixels to.
    xmin : float
        The minimum x value of the grid.
    ymin : float
        The minimum y value of the grid.
    res : float
        The resolution of the grid.
    outer : bool, optional
        If True, draw the outer circle. Otherwise, draw a filled circle.

    Returns
    -------
    None
    """
    center = ((np.asarray(xy) - (xmin, ymin)) / res).astype(int)
    radius = int(r / res) + 1 if outer else int(r / res)
    color = bool(value)
    thickness = 1 if outer else -1
    cv2.circle(img, center, radius, color, thickness)


class OdorTargetOnlyArena(ObstacleOdorArena):
    def __init__(
        self,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        seed=None,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)

        target_position = _get_random_target_position(
            distance_range=target_distance_range,
            angle_range=target_angle_range,
            rng=rng,
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


class ScatteredPillarsArena(ObstacleOdorArena):
    """
    An arena with scattered pillars and a target marker.

    This class generates an arena with randomly placed pillars and a target marker.
    The target marker is placed at a random position within a specified distance
    and angle range. Pillars are placed randomly while maintaining a minimum
    separation from the target, the fly, and other pillars.

    Parameters
    ----------
    target_distance_range : tuple[float, float], optional
        Range of distances from the origin for the target marker. Default is (29, 31).
    target_angle_range : tuple[float, float], optional
        Range of angles (in radians) for the target marker. Default is (-π, π).
    target_clearance_radius : float, optional
        Minimum clearance radius around the target marker. Default is 4.
    target_marker_size : float, optional
        Size of the target marker. Default is 0.3.
    target_marker_color : tuple[float, float, float, float], optional
        RGBA color of the target marker. Default is (1, 0.5, 14/255, 1).
    pillar_height : float, optional
        Height of the pillars. Default is 3.
    pillar_radius : float, optional
        Radius of the pillars. Default is 0.3.
    pillars_minimum_separation : float, optional
        Minimum separation between pillars. Default is 6.
    fly_clearance_radius : float, optional
        Minimum clearance radius around the fly. Default is 4.
    seed : int or None, optional
        Seed for the random number generator. Default is None.
    **kwargs : dict
        Additional keyword arguments passed to the parent class.

    Attributes
    ----------
    odor_source : np.ndarray
        Position of the odor source.
    marker_colors : np.ndarray
        Colors of the markers.
    peak_odor_intensity : np.ndarray
        Peak intensity of the odor.
    obstacle_positions : np.ndarray
        Positions of the pillars.
    obstacle_radius : float
        Radius of the pillars.
    obstacle_height : float
        Height of the pillars.
    terrain : FlatTerrain
        Terrain of the arena.
    marker_size : float
        Size of the target marker.
    """

    def __init__(
        self,
        target_distance_range=(29, 31),
        target_angle_range=(-np.pi, np.pi),
        target_clearance_radius=4,
        target_marker_size=0.3,
        target_marker_color=(1, 0.5, 14 / 255, 1),
        pillar_height=3,
        pillar_radius=0.3,
        pillars_minimum_separation=6,
        fly_clearance_radius=4,
        seed=None,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)

        target_position = _get_random_target_position(
            distance_range=target_distance_range,
            angle_range=target_angle_range,
            rng=rng,
        )

        pillar_positions = self._get_pillar_positions(
            target_position=target_position,
            target_clearance_radius=target_clearance_radius,
            pillar_radius=pillar_radius,
            pillars_minimum_separation=pillars_minimum_separation,
            fly_clearance_radius=fly_clearance_radius,
            rng=rng,
        )

        super().__init__(
            terrain=FlatTerrain(ground_alpha=0),
            obstacle_positions=pillar_positions,
            obstacle_radius=pillar_radius,
            obstacle_height=pillar_height,
            odor_source=np.array([[*target_position, 1]]),
            peak_odor_intensity=np.array([[1, 0]]),
            marker_colors=np.array([target_marker_color]),
            marker_size=target_marker_size,
            **kwargs,
        )

    @staticmethod
    def _get_pillar_positions(
        target_position: tuple[float, float],
        target_clearance_radius: float,
        pillar_radius: float,
        pillars_minimum_separation: float,
        fly_clearance_radius: float,
        rng: np.random.Generator,
        res: float = 0.05,
    ):
        """Generate random pillar positions.

        Parameters
        ----------
        target_position : tuple[float, float]
            The target x and y coordinates.
        target_clearance_radius : float
            The radius of the area around the target that should be clear of pillars.
        pillar_radius : float
            The radius of the pillars.
        pillars_minimum_separation : float
            Minimum separation between pillars.
        fly_clearance_radius : float
            The radius of the area around the fly that should be clear of pillars.
        rng : np.random.Generator
            The random number generator.
        res : float, optional
            The resolution of the grid. Default is 0.05.

        Returns
        -------
        np.ndarray
            The positions of the pillars in the form of [[x1, y1], [x2, y2], ...].
        """
        pillar_clearance_radius = pillar_radius * 2 + pillars_minimum_separation
        target_clearance_radius = target_clearance_radius + pillar_radius
        fly_clearance_radius = fly_clearance_radius + pillar_radius

        target_position = np.asarray(target_position)
        distance = np.linalg.norm(target_position)
        xmin = ymin = -distance
        xmax = ymax = distance
        n_cols = int((xmax - xmin) / res)
        n_rows = int((ymax - ymin) / res)
        im1 = np.zeros((n_rows, n_cols), dtype=np.uint8)
        im2 = np.zeros((n_rows, n_cols), dtype=np.uint8)

        _circ(im1, (0, 0), distance, 1, xmin, ymin, res)
        _circ(im1, (0, 0), fly_clearance_radius, 0, xmin, ymin, res)
        _circ(im1, target_position, target_clearance_radius, 0, xmin, ymin, res)

        pillars_xy = [target_position / 2]
        _circ(im1, pillars_xy[0], pillar_clearance_radius, 0, xmin, ymin, res)
        _circ(
            im2, pillars_xy[0], pillar_clearance_radius, 1, xmin, ymin, res, outer=True
        )

        while True:
            argwhere = np.argwhere(im1 & im2)
            try:
                p = argwhere[rng.choice(len(argwhere)), ::-1] * res + (xmin, ymin)
            except ValueError:
                break
            pillars_xy.append(p)
            _circ(im1, p, pillar_clearance_radius, 0, xmin, ymin, res)
            _circ(im2, p, pillar_clearance_radius, 1, xmin, ymin, res, outer=True)

        return np.array(pillars_xy)


class LoomingBallArena(FlatTerrain):
    """
    Simulates a looming ball scenario where a ball approaches a fly entity from different angles
    with Poisson-distributed spawning events.
    """

    def __init__(
        self,
        timestep,
        fly,
        ball_radius=1.0,
        ball_approach_vel=50,
        ball_approach_start_radius=20,
        ball_overshoot_dist=5,
        looming_lambda=1.0,
        seed=0,
        approach_angles=np.array([np.pi / 4, 3 * np.pi / 4]),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.fly = fly
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dt = timestep
        self.ball_radius = ball_radius

        self._setup_probabilities(looming_lambda)
        self._setup_trajectory_params(
            ball_approach_vel, ball_approach_start_radius, ball_overshoot_dist
        )
        self._setup_ball_heights()

        self.ball_approach_angles = approach_angles
        self.is_looming = False
        self.ball_traj_advancement = 0

        self._setup_velocity_buffer()
        self.add_ball(ball_radius)

    def _setup_probabilities(self, looming_lambda):
        """Initialize Poisson probability settings."""
        self.p_no_looming = np.exp(-looming_lambda * self.dt)

    def _setup_trajectory_params(self, vel, start_radius, overshoot_dist):
        """Precompute trajectory parameters for the looming ball."""
        self.ball_approach_vel = vel
        self.ball_approach_start_radius = start_radius
        self.overshoot_dist = overshoot_dist

        interception_time = start_radius / vel
        self.n_interception_steps = int(interception_time / self.dt)
        self.n_overshoot_steps = int((overshoot_dist / vel) / self.dt)

        self.ball_trajectory = np.zeros(
            (self.n_interception_steps + self.n_overshoot_steps, 2)
        )

    def _setup_ball_heights(self):
        """Calculate ball positions for visible and resting states."""
        self.ball_rest_height = 10.0
        self.ball_act_height = self.ball_radius + self._get_max_floor_height()

    def _setup_velocity_buffer(self):
        """Setup a fixed-length FIFO buffer for fly velocity estimation."""
        self.vel_buffer_size = 500
        self.fly_velocities = np.full((self.vel_buffer_size, 2), np.nan)
        self.fly_velocities_idx = 0

    def add_ball(self, ball_radius):
        """Add the ball to the scene with joints and geometry."""
        self.ball_body = self.root_element.worldbody.add(
            "body",
            name="ball_mocap",
            pos=[0, 0, self.ball_rest_height],
            mocap=True,
        )

        self.ball_geom = self.ball_body.add(
            "geom",
            name="ball",
            type="sphere",
            size=[ball_radius],
            rgba=[1, 0, 0, 0],
            density=1.0,
        )

    def spawn_entity(self, entity, rel_pos, rel_angle):
        """Spawn the fly and setup collision pairs."""
        super().spawn_entity(entity, rel_pos, rel_angle)
        self._add_contacts()

    def _add_contacts(self):
        """Add contact pairs between the ball and key fly body parts."""
        ball_geom_name = self.ball_geom.name
        for animat_geom_name in ["Head", "Thorax", "A1A2", "A3", "A4", "A5", "A6"]:
            self.root_element.contact.add(
                "pair",
                name=f"{ball_geom_name}_{self.fly.name}_{animat_geom_name}",
                geom1=f"{self.fly.name}/{animat_geom_name}",
                geom2=ball_geom_name,
                solimp=[0.995, 0.995, 0.001],
            )

    def set_ball_trajectory(self, start_pts, end_pts):
        """Generate a linear trajectory from start to end."""
        self.ball_trajectory = np.linspace(
            start_pts, end_pts, self.n_interception_steps + self.n_overshoot_steps
        )

    def make_ball_visible(self, physics):
        physics.bind(self.ball_geom).rgba[3] = 1

    def make_ball_invisible(self, physics):
        physics.bind(self.ball_geom).rgba[3] = 0

    def move_ball(self, physics, x, y, z):
        """Move ball to the desired location using joint positions."""
        physics.bind(self.ball_body).mocap_pos = np.array([x, y, z])

    def _get_mean_fly_velocity(self):
        """Compute average fly velocity from buffer."""
        return np.nanmean(self.fly_velocities, axis=0)

    def _should_trigger_ball(self):
        """Check if ball should start looming based on Poisson process."""
        return self.random_state.rand() > self.p_no_looming and not self.is_looming

    def _compute_trajectory_from_fly(self, fly_pos, fly_vel, fly_or_vec):
        """Generate start/end points of the ball trajectory based on fly state."""
        fly_roll = np.arctan2(fly_or_vec[1], fly_or_vec[0])
        approach_side = self.random_state.choice([-1, 1])
        rel_angles = self.ball_approach_angles * approach_side + fly_roll
        start_angle = self.random_state.uniform(low=rel_angles[0], high=rel_angles[1])

        interception_pos = fly_pos + fly_vel * self.n_interception_steps * self.dt
        start_pos = interception_pos + self.ball_approach_start_radius * np.array(
            [np.cos(start_angle), np.sin(start_angle)]
        )
        end_pos = interception_pos - self.overshoot_dist * np.array(
            [np.cos(start_angle), np.sin(start_angle)]
        )
        return start_pos, end_pos, interception_pos, start_angle

    def step(self, dt, physics):
        """Main loop: updates ball state, triggers looming events, and moves the ball."""
        # Update fly velocity buffer
        fly_vel = physics.bind(self.fly._body_sensors[1]).sensordata[:2].copy()
        self.fly_velocities[self.fly_velocities_idx % self.vel_buffer_size] = fly_vel
        self.fly_velocities_idx += 1

        if self._should_trigger_ball():
            self.is_looming = True
            self.ball_traj_advancement = 0
            self.make_ball_visible(physics)

            fly_pos = physics.bind(self.fly._body_sensors[0]).sensordata[:2].copy()
            fly_or_vec = physics.bind(self.fly._body_sensors[4]).sensordata.copy()
            fly_vel_mean = self._get_mean_fly_velocity()

            (
                start_pts,
                end_pts,
                interception_pos,
                angle,
            ) = self._compute_trajectory_from_fly(fly_pos, fly_vel_mean, fly_or_vec)

            self.set_ball_trajectory(start_pts, end_pts)
            self.move_ball(physics, *start_pts, self.ball_act_height)
            self.ball_traj_advancement += 1

            # Optional: visualize
            # self._plot_trajectory_debug(fly_pos, fly_vel_mean, interception_pos, start_pts, fly_or_vec)

        elif self.is_looming:
            self._advance_ball(physics)
        else:
            self.move_ball(physics, 0, 0, self.ball_rest_height)

    def _advance_ball(self, physics):
        """Advance the ball along its trajectory."""
        pos = self.ball_trajectory[self.ball_traj_advancement]
        self.move_ball(physics, pos[0], pos[1], self.ball_act_height)
        self.ball_traj_advancement += 1

        if (
            self.ball_traj_advancement
            >= self.n_interception_steps + self.n_overshoot_steps
        ):
            self.is_looming = False
            self.make_ball_invisible(physics)
            self.move_ball(physics, 0, 0, self.ball_rest_height)

    def _plot_trajectory_debug(
        self, fly_pos, fly_vel, intercept_pos, start_pos, orientation_vec
    ):
        """Visualize trajectory for debugging."""
        plt.scatter(fly_pos[0], fly_pos[1], label="fly pos", s=5)
        plt.scatter(
            intercept_pos[0], intercept_pos[1], label="fly interception pos", s=5
        )
        plt.plot(
            self.ball_trajectory[:, 0],
            self.ball_trajectory[:, 1],
            label="ball trajectory",
        )
        plt.scatter(start_pos[0], start_pos[1], label="ball start pos", s=5)
        plt.arrow(
            fly_pos[0], fly_pos[1], fly_vel[0], fly_vel[1], head_width=0.5, fc="blue"
        )
        plt.arrow(
            intercept_pos[0],
            intercept_pos[1],
            orientation_vec[0],
            orientation_vec[1],
            head_width=0.5,
            fc="green",
        )
        plt.legend()
        plt.show()

    def reset(self, physics, seed=None):
        """Reset the environment and optionally reseed."""
        if seed is not None:
            self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        self.is_looming = False
        self.ball_traj_advancement = 0
        self.make_ball_invisible(physics)
        self.ball_trajectory = np.zeros((self.n_interception_steps, 2))
        self.move_ball(physics, 0, 0, self.ball_rest_height)


class HierarchicalArena:
    pass


class FoodToNestArena:
    pass
