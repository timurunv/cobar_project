import numpy as np
from cobar_miniproject.base_controller import BaseController
from .utils import get_cpg, step_cpg
from .olfaction import compute_olfaction_control_signal

# python run_simulation.py ./submission/ --level 0 --progress
#python3 run_simulation.py ./submission/ --level 0 --progress

class Controller(BaseController):
    def __init__(
        self,
        timestep=1e-4,
        seed=0,
        action = np.ones((2,)),
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.action = action

    def get_actions(self, obs):
        
        # olfaction
        weight_olfaction = 10.0 # ideas : weight dependent on intensity (love blindness), internal states
        self.action += weight_olfaction * compute_olfaction_control_signal(obs)

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=self.action,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
