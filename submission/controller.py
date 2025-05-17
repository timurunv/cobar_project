import os 
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from flygym.vision.retina import Retina

from cobar_miniproject.base_controller import Action, BaseController, Observation

from .utils import get_cpg, step_cpg, compute_optic_flow, prepare_fly_vision
from .olfaction import compute_olfaction_turn_bias, compute_stationary_olfaction_bias
from .pillar_avoidance import compute_pillar_avoidance
from .ball_avoidance import is_ball
from .proprioception import predict_roll_change, get_stride_length_instantaneous, extract_proprioceptive_variables_from_stride
from .tests import test_heading, test_proprio, save_trajectories_for_path_integration_model
from .proprioception import get_stride_length

TEST_PATH = Path('outputs/test_heading')
TEST_PATH.mkdir(parents=True, exist_ok=True)

# python run_simulation.py --level 4 --gen_trajectories --savevid --saveplot --max-steps 30000
# python run_simulation.py --level 2 --output-dir outputs/test_heading --saveplot --max-steps 2000
# python run_simulation.py --level 0 --max-steps 2000
# python3 run_simulation.py --level 0 --max-steps 2000

class Controller(BaseController):
    def __init__(
        self,  
        timestep=1e-4,
        seed=0,
        weight_olfaction=0.5, # ideas : weight dependent on intensity (love blindness), internal states
        weight_pillar_avoidance=0.5, # ideas : weight dependent on intensity (love blindness), internal states
        obj_threshold=0.3, # threshold for object detection 
        ball_threshold=0.01, # threshold for ball detection
        seed_sim = 0,
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
        
        # backwards walking
        self.going_backward = False
        self.counter_backwards = 0
        self.velocities = []

        # Proprioception
        self.heading_angle = 0 # initial heading angle in fly-centric
        self.heading_angles = [] 
        self.path_int_buffer = [] 
        self.position = np.array([0, 0]) # initial position in fly_centric space
        self.estimated_orient_change = []
        self.vision_window_length = 2 # updating every 2 vision update steps
        self.proprio_window_length = 1200 # updating every stance cycle of the fly (0.12[s]/1e-4[s/step])
        self.vision_buffer = []
        self.counter_vision_buffer = -1
        self.last_end_effector_pos = None
        self.displacement_x = np.array([])
        self.displacement_y = np.array([])
        self.stride_lengths = []
        self.tests_counter = 0
        self.seed_sim = seed_sim

        self.fly_roll_hist = [] # TODO remove when finished testing 

        

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

    def _update_internal_heading(self, obs):
        del self.vision_buffer[0] # remove first entry
        self.vision_buffer.append(obs["vision"]) # update with latest vision
        delta_roll_pred = predict_roll_change(self.vision_buffer, n_top_pixels=5) # 5 is the best empirically found value
        self.estimated_orient_change.append(delta_roll_pred) # append to memory
        
        cum_estimated_orient_change = np.cumsum(self.estimated_orient_change) # assuming dt = 1, ie integrating over the internal clock of the fly
        self.heading_angle += cum_estimated_orient_change[-1] # take latest and update it 
        self.heading_angles.append(self.heading_angle) # TODO remove quand implémenté la distance


    def _compute_proprioceptive_variables(self):
        """
        This function computes the proprioceptive variables for the fly.
        It uses the stride length and contact forces to calculate the proprioceptive heading and distance signals.
        It gets its data from self.path_int_buffer.
        This implementation assumes that the initial heading is zero.
        """
        # heading = np.array([step['heading'] for step in self.path_int_buffer]) # pitch roll yaw in -> 
        contact_forces = np.array([step['contact_forces'] for step in self.path_int_buffer])
        stride_lenghts = np.array([step['stride_length'] for step in self.path_int_buffer])

        proprio_heading_pred, proprio_distance_pred = extract_proprioceptive_variables_from_stride(stride_lenghts, contact_forces, self.proprio_window_length)
        
        # Remove previous stance cycle
        del self.path_int_buffer[:self.proprio_window_length]

        return proprio_heading_pred, proprio_distance_pred

    def get_actions(self, obs, generate_trajectories=False):
        visual_features, rgb_features = self._process_visual_observation(obs)

        ball_alert = is_ball(rgb_features, self.ball_threshold)
        
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
            
            self.counter_backwards = self.counter_backwards + 1
            if self.counter_backwards > 2000 :
                if self.counter_backwards % 5  == 0:
                    self.velocities.append(obs['velocity'])
                    
                if self.counter_backwards % 1000 == 0 :
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

        #Vision-based path integration
        if obs.get("vision_updated", False):
            if self.counter_vision_buffer == -1: # initialization
                self.vision_buffer.append(obs["vision"]) ; self.vision_buffer.append(obs["vision"]) # append twice to fill the buffer
                self.counter_vision_buffer = 0

            self.counter_vision_buffer += 1

            # TEST HEADING # TODO retest when pillar avoidance is working
            # self.tests_counter += 1
            # test_heading(self.tests_counter, obs, self.vision_buffer[-1], self.fly_roll_hist, self.estimated_orient_change, debug= True)
    
            if self.counter_vision_buffer >= self.vision_window_length: # append only every vision_window_length steps
                self.counter_vision_buffer = 0
                self._update_internal_heading(obs)
                self.fly_roll_hist.append(self.path_int_buffer[-1]["heading"])

        # _____________________________________
        # Proprioceptive-based path integration
        # _____________________________________
        # compute stride length at each step
        stride_length, self.last_end_effector_pos = get_stride_length_instantaneous(obs["end_effectors"], self.last_end_effector_pos)
        self.path_int_buffer.append({'velocity': obs['velocity'], 'heading' : obs["heading"], 'contact_forces': obs['contact_forces'], 'stride_length' : stride_length, 'end_effectors' : obs["end_effectors"]}) # TODO maybe remove heading if computed differently
        self.stride_lengths.append(stride_length)
        version_window = False

        if version_window:
            if len(self.path_int_buffer) >= 2 * self.proprio_window_length: # update proprioceptive 
                proprio_heading_pred, proprio_distance_pred = self._compute_proprioceptive_variables()
                headings_obs = np.array([obs['heading'] for obs in self.path_int_buffer])
                # np.save(TEST_PATH / 'headings_obs.npy', headings_obs)
                headings_obs = np.unwrap(headings_obs, discont=np.pi)
                displacement_diff_x_pred = np.cos(headings_obs) * proprio_distance_pred
                displacement_diff_y_pred = np.sin(headings_obs) * proprio_distance_pred
                self.displacement_x = np.hstack([self.displacement_x, displacement_diff_x_pred])
                self.displacement_y = np.hstack([self.displacement_y, displacement_diff_y_pred])

                # self.tests_counter += 1
                # test_proprio(self.tests_counter, proprio_heading_pred, proprio_distance_pred)

                # speed = np.mean([obs['velocity'] for obs in self.path_int_buffer]) # average speed over the last window_length steps
                # dt = 1
                # self.position += speed * heading_vector * dt  # update position in world space

    
        if obs.get('reached_odour', False): # finished level -> return home
            print("Odour detected")
 
            # TEST ############################
            if not version_window:
                # end_effectors = np.array([step['end_effectors'] for step in self.path_int_buffer])
                contact_forces = np.array([step['contact_forces'] for step in self.path_int_buffer])
                heading = np.array([step['heading'] for step in self.path_int_buffer])
                heading = np.unwrap(heading, discont=np.pi)

                stride_length = np.array(self.stride_lengths)

                proprioceptive_heading_pred, proprioceptive_dist_pred = extract_proprioceptive_variables_from_stride(
                    stride_length, contact_forces, window_len=self.proprio_window_length
                )
                if generate_trajectories:
                    save_trajectories_for_path_integration_model(distance_pred = proprioceptive_dist_pred, heading_pred_optic = self.heading_angles, heading_pred= proprioceptive_heading_pred, seed=self.seed_sim, fly_roll=self.fly_roll_hist)
            

                # Integrate displacement
                displacement_diff_pred = proprioceptive_dist_pred

                heading = heading[:displacement_diff_pred.shape[0]]
                displacement_diff_x_pred = displacement_diff_pred * np.cos(heading)
                displacement_diff_y_pred = displacement_diff_pred * np.sin(heading)

                self.displacement_x = displacement_diff_x_pred
                self.displacement_y = displacement_diff_y_pred

            pos_x_pred = np.cumsum(self.displacement_x / self.proprio_window_length)
            pos_y_pred = np.cumsum(self.displacement_y / self.proprio_window_length)
            pos_pred = np.stack([pos_x_pred, pos_y_pred], axis=1)
            pos_pred = np.concatenate([np.full((self.proprio_window_length, 2), np.nan), pos_pred], axis=0) # pad with nan before the first window

            
            # test_proprio(100, None, None, pos_pred[:,0],  pos_pred[:, 1]) # to save the variables to memory


            # return_vector = pos_pred[-1] - pos_pred[0] # vector from start to end of the path
            # return_angle = np.arctan2(pos_pred[-1, 1], pos_pred[-1, 0]) # angle of the return vector



            # return_angle = np.arctan2(return_vector[1], return_vector[0])
            # adapt the action to the return vector based on x and y direction covered
            # self.action = 

            self.quit = True
        
        
                
        
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
