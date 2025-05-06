import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from .olfaction import compute_olfaction_turn_bias
from .pillar_avoidance import compute_pillar_avoidance
from flygym.vision.retina import Retina
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
#python run_simulation.py --level 0 --max-steps 2000
#python3 run_simulation.py --level 0 --max-steps 2000

class Controller(BaseController):
    def __init__(
        self,  
        timestep=1e-4,
        seed=0,
        weight_olfaction=0.5, # ideas : weight dependent on intensity (love blindness), internal states
        weight_pillar_avoidance=0.5, # ideas : weight dependent on intensity (love blindness), internal states
        obj_threshold=0.3, # threshold for object detection 
    ):
        from flygym.examples.locomotion import PreprogrammedSteps

        super().__init__()
        self.quit = False
        self.cpg_network = get_cpg(timestep=timestep, seed=seed)
        self.preprogrammed_steps = PreprogrammedSteps()
        self.action = np.ones((2,))
        self.weight_olfaction = weight_olfaction

        self.retina = Retina()
        self.obj_threshold = obj_threshold
        self.weight_pillar_avoidance = weight_pillar_avoidance
        self.coms = np.empty((self.retina.num_ommatidia_per_eye, 2))
        for i in range(self.retina.num_ommatidia_per_eye):
            mask = self.retina.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)
        
        self.velocities = []
        self.counter = 0
        self.going_backward = False

    def _process_visual_observation(self, raw_obs):
        features = np.zeros((2, 3))
        half_idx = np.unique(self.retina.ommatidia_id_map[220:], return_counts=False) #TODO maybe increase pr pas que ca bloque
        raw_obs["vision"][:, half_idx[:-1], :] = True
        for i, ommatidia_readings in enumerate(raw_obs["vision"]): #row_obs["vision"] of shape (2, 721, 2)
            is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold # shape (721, )
            is_obj_coords = self.coms[is_obj] # ommatidias in which object seen (nb ommatidia with object, 2 ), 2 for x and y coordinates
            if is_obj_coords.shape[0] > 0: #if there are ommatidia with object seen
                features[i, :2] = is_obj_coords.mean(axis=0) # mean of each x and y coordinate from ommatida with object seen (2, ) --> center of object seen
            features[i, 2] = is_obj_coords.shape[0] # number of ommatidia with object seen (1, ) --> area of object seen
        features[:, 0] /= self.retina.nrows  # normalize y_center
        features[:, 1] /= self.retina.ncols  # normalize x_center
        features[:, 2] /= self.retina.num_ommatidia_per_eye  # normalize area
        return features.ravel() # shape (6,) --> used in compute_pillar_avoidance


    def get_actions(self, obs):

        #Vision
        self.counter = self.counter + 1
        visual_features = self._process_visual_observation(obs)
        proximity_weight = np.clip(max(visual_features[2], visual_features[5]), 0, 0.2) / 0.2
        vision_action, object_detected = compute_pillar_avoidance(visual_features)
        if self.counter > 5000 :
            if self.counter % 5  == 0:
                self.velocities.append(obs['velocity'])
            if self.counter % 1000 == 0 :
                velocities_array = np.array([np.array(v) for v in self.velocities])
                smoothed_velocity = gaussian_filter1d(velocities_array[:,0], sigma=15) #take forward velocity component
                avrg_velocity = np.mean(smoothed_velocity)
                
                if avrg_velocity < 5 and avrg_velocity > -5:
                    self.going_backward = True
                    self.velocities = []
                else : 
                    self.going_backward = False
            if self.going_backward : 
                print("je suis le roi du monde")
                    
                
        #if self.counter == 50000:
        #    np.save("/Users/theolacroix/Desktop/MA4_EPFL_Cobar/cobar_project/outputs/caca", self.velocities)
        #If object right in front, little turn towards olfaction (#TODO: or maybe add a dynamic weight to olfaction and vision?)
        # if self.action[0] == self.action[1] and object_detected:
        #     olf_action = np.ones((2,))
        #     olf_action += compute_olfaction_turn_bias(obs) # it will subtract from either side
        #     self.action += olf_action
        #     self.action /= 2     

        #TODO: for the ball, if in front, normal avoidance, if on side and of certain size, increase overall speed   
        
        #Olfaction
        if not object_detected:
            self.action = np.ones((2,))
            self.action += compute_olfaction_turn_bias(obs) # it will subtract from either side

        olfaction_action = np.ones((2,)) + compute_olfaction_turn_bias(obs)
        if self.going_backward : 
            self.action = np.array([-1.0, -1.0])
        if not self.going_backward : 
            self.action = (1 - proximity_weight) * olfaction_action + proximity_weight * vision_action

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=self.action,
        )
        print(self.action)
        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
