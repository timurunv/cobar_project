import numpy as np
from cobar_miniproject.base_controller import BaseController
from .utils import get_cpg, step_cpg
from .olfaction import compute_olfaction_turn_bias
#python run_simulation.py --level 0 --max-steps 2000
#python3 run_simulation.py --level 0 --max-steps 2000

class Controller(BaseController):
    def __init__(
        self,  
        timestep=1e-4,
        seed=0,
        weight_olfaction=1.0, # ideas : weight dependent on intensity (love blindness), internal states
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.action = np.ones((2,))
        self.weight_olfaction = weight_olfaction

    def get_actions(self, obs):
        self.action = np.ones((2,)) # at each loop ? or cumulative so it turns faster ? maybe its the same as the weight idea
        
        # olfaction
        self.action += compute_olfaction_turn_bias(obs) # it will subtract from either side

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
