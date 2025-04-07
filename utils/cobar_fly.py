import flygym
import numpy as np

class CobarFly(flygym.Fly):
    def __init__(self, debug=False, **kwargs):
        super().__init__(
            contact_sensor_placements=[
                f"{side}{pos}{segment}"
                for side in "LR"
                for pos in "FMH"
                for segment in ["Tarsus5"]
            ],
            self_collisions="none",
            floor_collisions=[
                f"{side}{pos}{dof}"
                for side in "LR"
                for pos in "FMH"
                for dof in ["Tarsus5"]
            ],
            xml_variant="seqik_simple",
            head_stabilization_model="thorax",
            neck_kp=1000,
            enable_adhesion=True,
            enable_olfaction=True,
            **kwargs,
        )
        self.debug = debug

        # add these joints to enable using head stabilisation models
        fly_head_body = self.model.worldbody.find("body", "Head")  # type: ignore
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head_yaw",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "1 0 0",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "0 1 0",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        fly_head_body.add(
            "joint",
            **{
                "name": "joint_Head_roll",
                "class": "nmf",
                "type": "hinge",
                "pos": "0 0 0",
                "axis": "0 0 1",
                "stiffness": str(self.neck_stiffness),
                "springref": "0.0",
                "damping": str(self.non_actuated_joint_damping),
                "frictionloss": "0.0",
            },
        )
        self.model.visual.map.zfar = 10  # type: ignore

    def get_observation(self, sim: flygym.Simulation):
        observation = super().get_observation(sim)

        # if we're running the fly in debug mode (for development) it will return all raw observations
        # otherwise, it will return only a reduced observation space with egocentric observations
        if self.debug:
            return observation

        end_effector_positions_relative = CobarFly.absolute_to_relative_pos(
            observation["end_effectors"][:, :2],
            observation["fly"][0, :2],
            observation["fly_orientation"],
        )
        velocities_relative = CobarFly.absolute_to_relative_pos(
            observation["fly"][1, :2],
            observation["fly"][0, :2],
            observation["fly_orientation"],
        )

        observation_to_return = {
            "joints": observation["joints"],
            "end_effectors": end_effector_positions_relative,
            "contact_forces": observation["contact_forces"],
            "heading": np.arctan2(
                observation["fly_orientation"][1], observation["fly_orientation"][0]
            ),
            "velocity": velocities_relative,
        }

        if self.enable_olfaction:
            observation_to_return["odor_intensity"] = observation["odor_intensity"]

        if self.enable_vision:
            observation_to_return["vision"] = observation["vision"]
            if self.render_raw_vision:
                observation_to_return["raw_vision"] = self.get_info()["raw_vision"]

        return observation_to_return

    def post_step(self, sim: flygym.Simulation):
        obs = self.get_observation(sim)
        reward = self.get_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.get_info()

        if self.enable_vision:
            vision_updated_this_step = sim.curr_time == self._last_vision_update_time
            self._vision_update_mask.append(vision_updated_this_step)
            info["vision_updated"] = vision_updated_this_step

        # Fly has flipped if the z component of the "up" cardinal vector is negative
        cardinal_vector_z = sim.physics.bind(self._body_sensors[6]).sensordata.copy()
        info["flip"] = cardinal_vector_z[2] < 0

        if self.head_stabilization_model is not None:
            # this is tracked to decide neck actuation for the next step
            self._last_observation = obs
            info["neck_actuation"] = self._last_neck_actuation

        return obs, reward, terminated, truncated, info

    @staticmethod
    def absolute_to_relative_pos(
        pos: np.ndarray, base_pos: np.ndarray, heading: np.ndarray
    ) -> np.ndarray:
        """
        This function converts an absolute position to a relative position
        with respect to the base position and heading of the fly.

        It is used to obtain the fly-centric end effector (leg tip) positions.

        """
        rel_pos = pos - base_pos.reshape(1, -1)
        heading = heading / np.linalg.norm(heading)
        angle = np.arctan2(heading[1], heading[0])
        rot_matrix = np.array(
            [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
        )
        pos_rotated = np.dot(rel_pos, rot_matrix.T)
        return pos_rotated