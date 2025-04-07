from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    def __init__(self, timestep: float = 1e-4):
        self.timestep = timestep

    @abstractmethod
    def __call__(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pass
