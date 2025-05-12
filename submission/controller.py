import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg, compute_optic_flow, prepare_fly_vision
from .olfaction import compute_olfaction_turn_bias
from .pillar_avoidance import compute_pillar_avoidance
from .proprioception import predict_roll_change, get_stride_length, extract_proprioceptive_variables_from_stride
from flygym.vision.retina import Retina
from .tests import test_heading, test_proprio

from pathlib import Path
TEST_PATH = Path('outputs/test_heading')
TEST_PATH.mkdir(parents=True, exist_ok=True)

import os 
#python run_simulation.py --level 1 --output-dir outputs/test_heading --saveplot --max-steps 2000
#python run_simulation.py --level 0 --max-steps 2000
#python3 run_simulation.py --level 0 --max-steps 2000

class Controller(BaseController):
    def __init__(
        self,  
        timestep=1e-4,
        seed=0,
        weight_olfaction=0.5, # ideas : weight dependent on intensity (love blindness), internal states
        weight_pillar_avoidance=0.5, # ideas : weight dependent on intensity (love blindness), internal states
        obj_threshold=0.5, # threshold for object detection
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
        
        # Proprioception
        self.heading_angle = 0 # initial heading angle in fly-centric
        self.heading_angles = [] # TODO remove quand implémenté
        self.obs_buffer = [] # 
        self.position = np.array([0, 0]) # initial position in fly_centric space
        self.estimated_orient_change = []
        self.vision_window_length = 2 # updating every 2 vision update steps
        self.proprio_window_length = 2400 # updating every 2 stance cycles of the fly (2*0.12[s]/1e-4[s/step])
        self.vision_buffer = []
        self.counter_vision_buffer = -1
        self.last_end_effector_pos = None

        self.tests_counter = 0

        self.fly_roll_hist = [] # TODO remove when finished testing
        

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
        return features.ravel() # shape (6,)

    def _update_internal_heading(self, obs):
        del self.vision_buffer[0] # remove first entry
        self.vision_buffer.append(obs["vision"]) # update with latest vision
        delta_roll_pred = predict_roll_change(self.vision_buffer, n_top_pixels=5) # 5 is the best empirically found value
        self.estimated_orient_change.append(delta_roll_pred) # append to memory
        
        # TODO voir si on fait ça ici
        cum_estimated_orient_change = np.cumsum(self.estimated_orient_change) # assuming dt = 1, ie integrating over the internal clock of the fly 
        self.heading_angle += cum_estimated_orient_change[-1] # take latest and update it 
        self.heading_angles.append(self.heading_angle) # TODO remove quand implémenté la distance


    def _compute_proprioceptive_variables(self):
        """
        This function computes the proprioceptive variables for the fly.
        It uses the stride length and contact forces to calculate the proprioceptive heading and distance signals.
        It gets its data from self.obs_buffer.
        """
        end_efector_pos = np.array([obs['end_effectors'][:,:2] for obs in self.obs_buffer])
        # heading = np.array([obs['heading'] for obs in self.obs_buffer]) # pitch roll yaw in -> 
        contact_forces = np.array([obs['contact_forces'] for obs in self.obs_buffer])

        stride_length, self.last_end_effector_pos = get_stride_length(end_effector_pos = end_efector_pos, last_end_effector_pos= self.last_end_effector_pos)
        proprio_heading_pred, proprio_distance_pred = extract_proprioceptive_variables_from_stride(stride_length, contact_forces)
        del self.obs_buffer[:int(self.proprio_window_length/2)] # remove previous stance cycle

        return proprio_heading_pred, proprio_distance_pred

    def get_actions(self, obs):
        
        #Vision
        if obs.get("vision_updated", False):
            visual_features = self._process_visual_observation(obs.copy())
            self.action, object_detected = compute_pillar_avoidance(visual_features)

            #Vision-based path integration
            if self.counter_vision_buffer == -1: # initialization
                self.vision_buffer.append(obs["vision"]) ; self.vision_buffer.append(obs["vision"]) # append twice to fill the buffer
                self.counter_vision_buffer = 0

            self.counter_vision_buffer += 1 #COMMENT TO OMMIT PATH INTEGRATION

            # TEST HEADING # TODO retest when pillar avoidance is working
            # self.tests_counter += 1
            # test_heading(self.tests_counter, obs, self.vision_buffer[-1], self.fly_roll_hist, self.estimated_orient_change, debug= True)
    
            if self.counter_vision_buffer >= self.vision_window_length: # append only every vision_window_length steps
                self.counter_vision_buffer = 0
                self._update_internal_heading(obs)
                self.fly_roll_hist.append(self.obs_buffer[-1]["heading"]) 

        else:
            object_detected = False
        
        #Olfaction
        if not object_detected:
            self.action = np.ones((2,))
            self.action += compute_olfaction_turn_bias(obs) # it will subtract from either side
        
        # Proprioceptive-based path integration
        self.obs_buffer.append({'velocity':obs['velocity'], 'heading' : obs["heading"], 'end_effectors' : obs["end_effectors"], 'contact_forces': obs['contact_forces']}) # TODO maybe remove heading if computed differently
        
        if len(self.obs_buffer) >= self.proprio_window_length: # update proprioceptive 
            proprio_heading_pred, proprio_distance_pred = self._compute_proprioceptive_variables()
            # print(proprio_distance_pred, proprio_heading_pred)
            # TEST distance
            self.tests_counter += 1
            # test_proprio(self.tests_counter, obs["end_effectors"], proprio_heading_pred, proprio_distance_pred)

            # heading_vector = np.array([np.cos(self.heading_angle), np.sin(self.heading_angle)])
            # speed = np.mean([obs['velocity'] for obs in self.obs_buffer]) # average speed over the last window_length steps
            # dt = 1
            # self.position += speed * heading_vector * dt  # update position in world space

            # TODO implement steps for distance




        if obs.get('reached_odour', False): # finished level -> return home
            print("Odour detected")
            # test_heading(self.tests_counter, obs, self.vision_buffer[-1], self.fly_roll_hist, self.estimated_orient_change, debug= True)
            # return_vector = -self.position


            # return_angle = np.arctan2(return_vector[1], return_vector[0])
            # adapt the action to the return vector based on x and y direction covered
            # self.action = 

            self.quit = True

        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=self.action,
        )

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
