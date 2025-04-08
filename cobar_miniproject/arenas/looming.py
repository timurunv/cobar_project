import numpy as np
from flygym.arena import FlatTerrain


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
        import matplotlib.pyplot as plt

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
