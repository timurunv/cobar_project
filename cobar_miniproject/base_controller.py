from abc import ABC, abstractmethod
import numpy as np
from typing import TypedDict


class Observation(TypedDict, total=False):
    joints: np.ndarray
    end_effectors: np.ndarray
    contact_forces: np.ndarray
    heading: float
    velocity: np.ndarray
    odor_intensity: np.ndarray
    vision: np.ndarray
    raw_vision: np.ndarray
    vision_updated: bool
    reached_odour: bool


class Action(TypedDict):
    """
    The fly's joint angles under the "joints" key and
    the adhesion state for each leg under the "adhesion" key.
    """

    joints: np.ndarray
    adhesion: np.ndarray


class BaseController(ABC):
    def __init__(self, **kwargs):
        """
        Initialize the CobarController with any necessary parameters.
        """

    @abstractmethod
    def get_actions(self, obs: Observation) -> Action:
        """
        The heart of your controller: goes from observations to joint angles.

        Parameters
        ----------
        obs: dict
            The observations from the environment. Namely
            - The angle of each of the fly's joints
            - The end effector positions in egocentric coordinates
            - The contact forces
            - The fly's absolute heading
            - The fly's velocity along x, y, z
            - The vision both raw and fly-like
            - The odour intensity in each antenna
            - Whether the fly's vision was updated in the last simulation step
            - Whether the fly has reached the odour source yet (used for the final level)

        Returns
        -------
        action: dict
            The fly's joint angles under the "joints" key
            The adhesion state under the "adhesion" key
        """

        action: Action = {
            "joints": np.zeros(42),
            "adhesion": np.zeros(6),
        }
        return action

    @abstractmethod
    def done_level(self, obs) -> bool:
        """
        For level 5 (path integration) - if the fly has returned back home
        after collecting the odour, return `True` here to stop the simulation.
        """
        pass

    @abstractmethod
    def reset(self, **kwargs):
        """
        Reset the controller to its initial state.
        """
        pass
