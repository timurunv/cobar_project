import numpy as np
from cobar_miniproject import BaseController
from .utils import get_cpg, step_cpg


class Controller(BaseController):
    def __init__(self):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.cpg_network = get_cpg(self.timestep)
        self.preprogrammed_steps = PreprogrammedSteps()

    def __call__(self, obs):
        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=np.array([1.0, 1.0]),
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }
