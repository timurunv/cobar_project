import numpy as np
from cobar_miniproject.base_controller import Action, BaseController, Observation
from .utils import get_cpg, step_cpg, compute_optic_flow, prepare_fly_vision
from .olfaction import compute_olfaction_turn_bias
from .pillar_avoidance import compute_pillar_avoidance
from flygym.vision.retina import Retina
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
        self.vision_window_length = 5 # updating every 5 simulation steps
        self.vision_buffer = np.zeros((2,self.retina.num_ommatidia_per_eye, 2, 2))
        self.counter_buffer = -1 # for efficiency

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

    def _predict_roll_change(self):
        pre_img = prepare_fly_vision(self.vision_buffer[0])
        post_img = prepare_fly_vision(self.vision_buffer[1])
        flow = compute_optic_flow(pre_img, post_img)
        mean_x_flow = np.mean(flow[..., 0])
        return mean_x_flow

    def _update_internal_heading(self, obs):
        self.vision_buffer = np.roll(self.vision_buffer, shift=-1, axis=0) # shift the buffer to the left
        self.vision_buffer[..., -1] = obs["vision"] # update with latest vision
        self.estimated_orient_change.append(self._predict_roll_change()) # append to memory
        
        # TODO voir si on fait ça ici
        cum_estimated_orient_change = np.cumsum(self.estimated_orient_change) # assuming dt = 1, ie integrating over the internal clock of the fly 
        self.heading_angle += cum_estimated_orient_change[-1] # take latest and update it 
        self.heading_angles.append[self.heading_angle] # TODO remove quand implémenté la distance

    def get_actions(self, obs):
        #Vision
        visual_features = self._process_visual_observation(obs)
        self.action, object_detected = compute_pillar_avoidance(visual_features)
        
        #Olfaction
        if not object_detected:
            self.action = np.ones((2,))
            self.action += compute_olfaction_turn_bias(obs) # it will subtract from either side

        #Proprioception #WORK IN PROGRESS
        if self.counter_buffer == -1: # initialize buffer to first vision observation
            self.vision_buffer[..., 0] = obs["vision"]
        # self.counter_buffer += 1 #UNCOMMENT TO RUN PROPRIOCEPTION PART

        if self.obs_buffer: # list not empty
            del self.obs_buffer[0] # remove first entry
        self.obs_buffer.append({'velocity':obs['velocity'], 'heading' : obs["heading"], 'end_effectors' : obs["end_effectors"]})
        # TODO QUESTION est ce que heading c'est pareil que fly_orientation ?

        if self.counter_buffer >= self.vision_window_length: # append only every vision_window_length steps
            self._update_internal_heading(obs)

            heading_vector = np.array([np.cos(self.heading_angle), np.sin(self.heading_angle)])
            speed = np.mean([obs['velocity'] for obs in self.obs_buffer]) # average speed over the last window_length steps
            dt = 1
            self.position += speed * heading_vector * dt  # update position in world space

            # TODO implement steps for distance and keep speed constant

            # ideas to approximate the speed and distance
            #    - use vision. probably best option but limited by the resolution of the ommatidia
            #       he sugggests using the shades of gray to increase the resolution
            #    - use descending drive and regress the speed from it
            #       seems not like a good idea
            #    - use proprioception to count the steps and get distance
            #
            #    => probably the best is to have a combination of all
        
        
        self.fly_roll_hist.append(obs["heading"]) # TODO remove when finished testing

        if obs.get('reached_odour', False): # finished level -> return home
            print("Odour detected")
            return_vector = -self.position
            return_angle = np.arctan2(return_vector[1], return_vector[0])
            # adapt the action to the return vector based on x and y direction covered
            # self.action = 

            # TODO pour l'instant ici mais idéalement de temps en temps
            # self.heading_angles
            # integrer le heading vu que en fly centric c'est juste aller de l'avant, donc projeter pour avoir la distance sur verticale 
            # sur x et y en world centric pour savoir la distance a faire sur l'hypothénuse

            cum_orientations_final = np.cumsum(self.estimated_orient_change)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(9, 3), tight_layout=True)
            ax.plot(cum_orientations_final, label="predicted")
            twin_ax = ax.twinx()
            twin_ax.plot(self.fly_roll_hist[::self.vision_window_length], label="true roll")
            plt.savefig('fly_roll_proprio_test.png')
            self.done_level(obs)

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
