import numpy as np
from flygym.vision import Retina
import cv2

#obs keys
#dict_keys(['joints', 'end_effectors', 'contact_forces', 'heading', 'velocity', 'odor_intensity', 'vision', 'raw_vision'])

# obs["end_effectors"].shape - (6, 2)  # diff than tuto # (6, 3)
# obs["contact_forces"].shape - (6, 3) # tuto (,6)
# obs["joints"].shape - (3, 42)
# obs["velocity"].shape - (1, 2)
# obs["odor_intensity"] -  [[0.0010888  0.00107006 0.0010581  0.00105298]
#                           [0.         0.         0.         0.        ]]

# obs["heading"] - int # tuto (,3)
# NO POSITION - (,3)



def get_cpg(timestep, seed=0):
    from flygym.examples.locomotion import CPGNetwork

    phase_biases = np.pi * np.array(
        [
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0],
        ]
    )
    coupling_weights = (phase_biases > 0) * 10

    return CPGNetwork(
        timestep=timestep,
        intrinsic_freqs=np.ones(6) * 36,
        intrinsic_amps=np.ones(6) * 6,
        coupling_weights=coupling_weights,
        phase_biases=phase_biases,
        convergence_coefs=np.ones(6) * 20,
        init_phases=None,
        init_magnitudes=None,
        seed=seed,
    )


def step_cpg(cpg_network, preprogrammed_steps, action):
    
    amps = np.repeat(np.abs(action[:, np.newaxis]), 3, axis=1).ravel()
    freqs = np.abs(cpg_network.intrinsic_freqs)
    freqs[:3] *= 1 if action[0] > 0 else -1
    freqs[3:] *= 1 if action[1] > 0 else -1
    cpg_network.intrinsic_amps = amps
    cpg_network.intrinsic_freqs = freqs
    cpg_network.step()

    joints_angles = []
    adhesion_onoff = []

    for i, leg in enumerate(preprogrammed_steps.legs):
        # get target angles from CPGs and apply correction
        my_joints_angles = preprogrammed_steps.get_joint_angles(
            leg,
            cpg_network.curr_phases[i],
            cpg_network.curr_magnitudes[i],
        )
        joints_angles.append(my_joints_angles)

        # get adhesion on/off signal
        my_adhesion_onoff = preprogrammed_steps.get_adhesion_onoff(
            leg, cpg_network.curr_phases[i]
        )

        # No adhesion in stumbling or retracted
        adhesion_onoff.append(my_adhesion_onoff)

    joint_angles = np.array(np.concatenate(joints_angles))
    adhesion_onoff = np.array(adhesion_onoff).astype(int)

    return joint_angles, adhesion_onoff


import matplotlib.pyplot as plt

def plot_trajectory(savepath, obs, obstacle_poz, odor_poz, obstacle_size = 2, odor_size = 0.1, save = True):
    plt.figure(figsize=(6,6), dpi=150)
    plt.plot([observation["fly"][0][0] for observation in obs], [observation["fly"][0][1] for observation in obs], label="Fly trajectory")
    if obstacle_poz.size !=0 : # if not empty
        plt.plot([op[0] for op in obstacle_poz], [op[1] for op in obstacle_poz], 'ko', markersize=obstacle_size*2, label='Obstacles')
        #TODO adapt marker size to obstacle size
    if odor_poz.size !=0 : 
        plt.plot([op[0] for op in odor_poz], [op[1] for op in odor_poz], 'ro', markersize=8, label='Odor Position')

    plt.title(str(savepath).split("\\")[-1].split('.')[0])
    plt.legend()
    plt.axis('equal')
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.savefig(savepath)
    plt.show()

def compute_optic_flow(img0, img1):
    img0 = (img0 * 255).astype(np.uint8)
    img1 = (img1 * 255).astype(np.uint8)
    
    flow = cv2.calcOpticalFlowFarneback(
        img0, img1, None, 0.5, 2, 3, 2, 5, 1.1, 0
    )
    return flow


def crop_hex_to_rect(visual_input):
    ommatidia_id_map = Retina().ommatidia_id_map
    rows = [np.unique(row) for row in ommatidia_id_map]
    max_width = max(len(row) for row in rows)
    rows = np.array([row for row in rows if len(row) == max_width])[:, 1:] - 1
    cols = [np.unique(col) for col in rows.T]
    min_height = min(len(col) for col in cols)
    cols = [col[:min_height] for col in cols]
    rows = np.array(cols).T
    return visual_input[..., rows, :].max(-1)


def prepare_fly_vision(two_eyes_vision, n_top_pixels=5):
    left_eye = two_eyes_vision[0]
    right_eye = two_eyes_vision[1]

    left_eye_square = crop_hex_to_rect(left_eye)
    right_eye_square = crop_hex_to_rect(right_eye)

    left_eye_top = left_eye_square[:n_top_pixels]
    right_eye_top = right_eye_square[:n_top_pixels]

    stacked_eye = np.vstack([left_eye_top, right_eye_top])

    return stacked_eye