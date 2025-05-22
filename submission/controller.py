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
from .proprioception import predict_roll_change, get_stride_length, extract_proprioceptive_variables_from_stride
from .tests import test_heading, test_proprio, save_trajectories_for_path_integration_model
from .proprioception import get_stride_lengths, load_proprioceptive_models

TEST_PATH = Path('outputs/test_heading')
TEST_PATH.mkdir(parents=True, exist_ok=True)

# python run_simulation.py --level 4 --gen_trajectories --savevid --saveplot --max-steps 40000
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
        self.heading_preds_optic = [] 
        self.path_int_buffer = [] 
        self.position = np.array([0, 0]) # initial position in fly_centric space
        self.estimated_orient_change = []
        self.vision_window_length = 1 # updating every vision update step
        self.proprio_window_length = 1200 # updating every stance cycle of the fly (0.12[s]/1e-4[s/step])
        self.vision_buffer = []
        self.counter_vision_buffer = -1
        self.last_end_effector_pos = None
        self.pos_x = None
        self.pos_y = None
        self.stride_lengths = []
        self.prop_heading_model, self.prop_disp_model, self.optic_heading_model, self.velocity_model = load_proprioceptive_models()

        self.seed_sim = seed_sim
        self.tests_counter = 0
        self.fly_roll_hist = [] 
        self.test_path_int_buffer = []

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
        delta_roll_pred = predict_roll_change(self.vision_buffer, n_top_pixels=8) # 8 is the best empirically found value
        self.estimated_orient_change.append(delta_roll_pred) # append to memory
        # cumulate the heading to keep track internally
        self.heading_angle += delta_roll_pred
        self.heading_preds_optic.append(self.heading_angle)

        # TEST path integration
        self.fly_roll_hist.append(obs["heading"])

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
        
        if ball_alert: #stop moving
            self.action = np.zeros((2,)) 
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

            # to test heading 
            # test_heading(self.tests_counter, obs, self.vision_buffer[-1], self.fly_roll_hist, self.estimated_orient_change, debug= True)
    
            if self.counter_vision_buffer >= self.vision_window_length: # append only every vision_window_length steps
                self.counter_vision_buffer = 0
                self._update_internal_heading(obs)

        # Proprioceptive-based path integration - compute stride length at each step and append values to buffer
        stride_length, self.last_end_effector_pos = get_stride_length(obs["end_effectors"], obs['heading'], self.last_end_effector_pos)
        self.stride_lengths.append(stride_length)
        self.path_int_buffer.append({'velocity': obs['velocity'], 'heading' : obs["heading"], 'contact_forces': obs['contact_forces'], 'stride_length' : stride_length, 'end_effectors' : obs["end_effectors"], 'drive' : self.action})
        
        if generate_trajectories:
            self.test_path_int_buffer.append({'heading' : obs["heading"], 'fly_x': obs["debug_fly"][0][0], 'fly_y' : obs["debug_fly"][0][1]})
    
        if obs.get('reached_odour', False): # finished level -> return home
            # under TESTING ############################
            contact_forces = np.array([step['contact_forces'] for step in self.path_int_buffer])
            heading = np.array([step['heading'] for step in self.path_int_buffer])
            heading = np.unwrap(heading, discont=np.pi)

            proprioceptive_heading_pred, proprioceptive_dist_pred = extract_proprioceptive_variables_from_stride(
                np.array(self.stride_lengths), contact_forces, window_len=self.proprio_window_length
            )
            
            if generate_trajectories:
                true_x = np.array([step['fly_x'] for step in self.test_path_int_buffer])
                true_y = np.array([step['fly_y'] for step in self.test_path_int_buffer])
                true_heading = np.array([step['heading'] for step in self.test_path_int_buffer])
                velocities = np.array([step['velocity'] for step in self.path_int_buffer])
                drives = np.array([step['drive'] for step in self.path_int_buffer])
                save_trajectories_for_path_integration_model(
                    x_true= true_x,y_true=true_y,heading_true=true_heading ,
                    distance_pred = proprioceptive_dist_pred, heading_pred_optic = self.heading_preds_optic, 
                    heading_pred= proprioceptive_heading_pred, seed=self.seed_sim, fly_roll=self.fly_roll_hist, 
                    velocity = velocities, drives = drives)
                print('true displacement', true_x[-1], true_y[-1], 'heading', true_heading[-1])

            heading_final = self.create_heading_final(proprioceptive_heading_pred, self.heading_preds_optic)
            
            velocity_x = None
            if velocity_x is None:
                heading_final = heading_final[self.proprio_window_length:] # remove first proprioceptive window since velocity not accurate in the displacement prediction
            disp_final = self.create_displacement_final(proprioceptive_dist_pred, velocities=velocity_x)

            displacement_diff_x_pred = disp_final.flatten() * np.cos(heading_final).flatten()
            displacement_diff_y_pred = disp_final.flatten() * np.sin(heading_final).flatten()

            pos_x_pred = np.cumsum(displacement_diff_x_pred / self.proprio_window_length)
            pos_y_pred = np.cumsum(displacement_diff_y_pred / self.proprio_window_length)
            
            self.pos_x = pos_x_pred[-1]
            self.pos_y = pos_y_pred[-1]
            print('pos_x_pred', pos_x_pred[-1], 'pos_y_pred', pos_y_pred[-1])
        
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



    def create_heading_final(self, proprioceptive_heading_pred, optic_heading_pred, optic_window = 100):
        """
        This function creates the final heading signal by combining the proprioceptive and optic flow signals.
        It uses the proprioceptive heading signal to fill in the gaps between optic flow updates.
        In the first self.proprio_window_length steps, it uses the optic flow signal only, as a step function.

        Parameters
        ----------
        proprioceptive_heading_pred : np.array
            The proprioceptive heading signal.
        optic_heading_pred : np.array
            The optic flow heading signal.
        optic_window : int
            The number of steps between optic flow updates (ie vision updated).
        """
        N = len(proprioceptive_heading_pred) + self.proprio_window_length
        heading_final = np.zeros(N)
        optic_indices = np.arange(0, N, optic_window)
        
        proprio_heading = self.prop_heading_model(np.array(proprioceptive_heading_pred).reshape(-1, 1)).flatten()
        # fill with zeros the first proprioceptive window so that later the delta is correct
        proprio_heading = np.concatenate([np.full((self.proprio_window_length), 0), proprio_heading], axis=0)/ self.proprio_window_length

        optic_headings_corrected = self.optic_heading_model.predict(np.array(optic_heading_pred).reshape(-1, 1)).flatten()
        
        for i, idx in enumerate(optic_indices): # every optic window set value to optic heading
            heading_final[idx] = optic_headings_corrected[i]

        # Fill in the initial proprioceptive window using optic flow only 
        for i in range(0, self.proprio_window_length - optic_window, optic_window):
            # carry forward last optic value (step-wise fill)
            for j in range(i + 1, i + optic_window):
                heading_final[j] = heading_final[j - 1]

        # Between optic updates use the proprioceptive heading
        for i in range(len(optic_indices) - 1):
            for j in range(optic_indices[i] + 1, optic_indices[i + 1]):
                heading_final[j] = heading_final[j - 1] + proprio_heading[j]

        # Handle the final segment after the last optic update
        for j in range(optic_indices[-1] + 1, N):
            heading_final[j] = heading_final[j - 1] + proprio_heading[j]
            
        return heading_final

    def create_displacement_final(self, proprioceptive_dist_pred, velocities):
        """
        This function creates the final displacement signal by combining the proprioceptive and velocity signals.
        It uses the velocity signal to fill the first proprioceptive window.
        Params
        ----------
        proprioceptive_dist_pred : np.array
            The proprioceptive distance signal.
        velocities : np.array
            The velocity signal.
        """
        if velocities is None: # do not use velocity
            additonal = 0
            N = len(proprioceptive_dist_pred)
            disp_final = np.zeros(N)
        else:
            additonal= self.proprio_window_length
            N = len(proprioceptive_dist_pred) + additonal
            disp_final = np.zeros(N)
            

            # Velocity for the first proprioceptive window
            smoothed_velocity = gaussian_filter1d(velocities, sigma=20)
            disp_pred_velocity = self.velocity_model.predict(np.array(smoothed_velocity).reshape(-1, 1)).flatten()
            disp_final[:self.proprio_window_length] = disp_pred_velocity[:self.proprio_window_length]

        proprio_disp_pred = self.prop_disp_model(proprioceptive_dist_pred)
        disp_final[additonal:] = proprio_disp_pred

        return disp_final