import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg
from .olfaction import compute_olfaction_turn_bias, compute_stationary_olfaction_bias
from .pillar_avoidance import compute_pillar_avoidance
from .ball_avoidance import is_ball
from flygym.vision.retina import Retina
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import math
#python run_simulation.py --level 0 --max-steps 2000
#python3 run_simulation.py --level 0 --max-steps 2000
#python3 run_simulation.py --level 4 --max-steps 30000 --saveplot --savevid --seed 19
#use numba for the next time

class Controller(BaseController):
    def __init__(
        self,  
        timestep=1e-4,
        seed=0,
        weight_olfaction=0.5, # ideas : weight dependent on intensity (love blindness), internal states
        weight_pillar_avoidance=0.5, # ideas : weight dependent on intensity (love blindness), internal states
        obj_threshold=0.3, # threshold for object detection 
        ball_threshold=0.01, # threshold for ball detection
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
        self.ball_threshold = ball_threshold
        self.weight_pillar_avoidance = weight_pillar_avoidance
        self.coms = np.empty((self.retina.num_ommatidia_per_eye, 2))
        for i in range(self.retina.num_ommatidia_per_eye):
            mask = self.retina.ommatidia_id_map == i + 1
            self.coms[i, :] = np.argwhere(mask).mean(axis=0)
        
        self.velocities = []
        self.counter = 0
        self.going_backward = False
        self.init_rotation_angle = True
        self.remaining_rotation = 0
        self.alpha_heading = 0
        self.distance_to_cover = 0
        self.alpha = 0

    def _process_visual_observation(self, raw_obs):
        features = np.zeros((2, 3))
        half_idx = np.unique(self.retina.ommatidia_id_map[250:], return_counts=False)
        raw_obs["vision"][:, half_idx[:-1], :] = True
        rgb_features = raw_obs["raw_vision"]

        #Check if object is close-by
        for i, ommatidia_readings in enumerate(raw_obs["vision"]): #row_obs["vision"] of shape (2, 721, 2)
            is_obj = ommatidia_readings.max(axis=1) < self.obj_threshold # shape (721, )
            is_obj_coords = self.coms[is_obj] # ommatidias in which object seen (nb ommatidia with object, 2 ), 2 for x and y coordinates
            if is_obj_coords.shape[0] > 0: #if there are ommatidia with object seen
                features[i, :2] = is_obj_coords.mean(axis=0) # mean of each x and y coordinate from ommatida with object seen (2, ) --> center of object seen
            features[i, 2] = is_obj_coords.shape[0] # number of ommatidia with object seen (1, ) --> area of object seen
        features[:, 0] /= self.retina.nrows  # normalize y_center
        features[:, 1] /= self.retina.ncols  # normalize x_center
        features[:, 2] /= self.retina.num_ommatidia_per_eye  # normalize area
        return features.ravel(), rgb_features # shape (6,) --> used in compute_pillar_avoidance


    def get_actions(self, obs):
        #Vision
        
        self.counter = self.counter + 1
        visual_features, rgb_features = self._process_visual_observation(obs)
        ball_alert = is_ball(rgb_features, self.ball_threshold)
        
        if(not(obs.get("reached_odour", False))): 
            if ball_alert:
                self.action = np.zeros((2,)) #stop moving
            else:
                #Pillar avoidance
                self.action, object_detected = compute_pillar_avoidance(visual_features)
                if self.action[0] == self.action[1] and object_detected:
                    self.action = compute_stationary_olfaction_bias(obs)   

                #Olfaction (only if no object or ball detected)
                if not object_detected:
                    self.action = np.ones((2,)) + compute_olfaction_turn_bias(obs)
                    
                if self.counter > 2000 :
                    if self.counter % 5  == 0:
                        self.velocities.append(obs['velocity'])
                        
                    if self.counter % 1000 == 0 :
                        velocities_array = np.array([np.array(v) for v in self.velocities])
                        smoothed_velocity = gaussian_filter1d(velocities_array[:,0], sigma=15) #take forward velocity component
                        avrg_velocity = np.mean(smoothed_velocity)
                        if avrg_velocity < 5 and avrg_velocity > -5:
                            self.going_backward = True
                        else : 
                            self.going_backward = False
                            
                        self.velocities = []
                    if self.going_backward : 
                        self.action = np.array([-1.0, -1.0])
            
        else:
            if(self.init_rotation_angle): 
                x_pred = -234
                y_pred = 554
                self.distance_to_cover = np.sqrt(x_pred**2 + y_pred**2)
                self.alpha = np.arctan2(-y_pred, -x_pred)
                self.init_rotation_angle = False
                
            error = self.alpha - obs["heading"]
            if error > math.pi:
                error -= 2 * math.pi
            elif error < -math.pi:
                error += 2 * math.pi

            Kp = 3.0 #proportional gain
            self.action = np.array([-1.0, 1.0]) * Kp * error 
            self.action[0] = np.clip(self.action[0], -1.0, 1.0)
            self.action[1] = np.clip(self.action[1], -1.0, 1.0)
            
            if(abs(error) < 0.1): 
                dt_per_step = 1e-4
                instantaneous_speed = np.linalg.norm(obs['velocity']) * 13
                dx_per_step = instantaneous_speed*dt_per_step
                self.distance_to_cover -= dx_per_step
                print("distance to cover", self.distance_to_cover)
                if(self.distance_to_cover < 0.001): 
                    self.quit = True
                self.action = np.array([1.0, 1.0])
        
        joint_angles, adhesion = step_cpg(
            cpg_network=self.cpg_network,
            preprogrammed_steps=self.preprogrammed_steps,
            action=self.action,
        )
        if ball_alert:
            adhesion = np.ones(6)

        return {
            "joints": joint_angles,
            "adhesion": adhesion,
        }

    def done_level(self, obs: Observation):
        return self.quit

    def reset(self, **kwargs):
        self.cpg_network.reset()
