from pathlib import Path
import numpy as np
import os
BASE_DIR = Path.cwd()
TEST_PATH = BASE_DIR / 'outputs' /'test_distance' # test_heading
TEST_PATH.mkdir(parents=True, exist_ok=True)
TRAJECTORY_PATH = Path('outputs/trajectories')

def test_heading(counter, obs, obs_vision, fly_roll_hist, estimated_orient_change, debug=False):
    # see jupyter notebook outputs/tests/test_heading.ipynb for the file where this is used
    if obs.get("vision_updated", False) :  #counter % 500 == 1: # at the 100+1 steps vision is updated
        np.save(TEST_PATH / f'vision_{counter}.npy', obs_vision)
    if obs.get('reached_odour', False):
        print('saving')
        cum_orientations_final = np.cumsum(estimated_orient_change)
        np.save(TEST_PATH / 'cum_orientations_final.npy', cum_orientations_final)
        if debug: 
            fly_roll_hist = np.unwrap(fly_roll_hist, discont=np.pi) # if debug mode
        np.save(TEST_PATH / 'fly_roll_hist.npy', fly_roll_hist)


def test_proprio(counter, proprio_heading_pred, proprio_distance_pred, pos_x,pos_y):
    # see jupyter notebook outputs/tests/test_proprio.ipynb for the file where this is used
    # if counter >300 and counter < 400:
    #     np.save(TEST_PATH / f'end_effectors_{counter}.npy', end_effectors)
    if counter % 100 == 0:
        if proprio_heading_pred is not None:
            np.save(TEST_PATH / f'proprio_heading_pred_{counter}.npy', proprio_heading_pred)
        if proprio_distance_pred is not None:
            np.save(TEST_PATH / f'proprio_distance_pred_{counter}.npy', proprio_distance_pred)
        if pos_x is not None:
            np.save(TEST_PATH / f'pos_x.npy', pos_x)
        if pos_y is not None:
            np.save(TEST_PATH / f'pos_y.npy', pos_y)



def save_trajectories_for_path_integration_model(distance_pred=None, heading_pred_optic=None, heading_pred=None, x_true=None, y_true=None, heading_true=None, fly_roll = None, seed = None):
    if distance_pred is not None:
        np.save(TRAJECTORY_PATH / f'pred_distance_{seed}.npy', distance_pred)
    if heading_pred_optic is not None:
        np.save(TRAJECTORY_PATH / f'pred_headingoptic_{seed}.npy', heading_pred_optic)
    if heading_pred is not None:
        np.save(TRAJECTORY_PATH / f'pred_heading_{seed}.npy', heading_pred)
    if x_true is not None:
        np.save(TRAJECTORY_PATH / f'true_x_{seed}.npy', x_true)
    if y_true is not None:
        np.save(TRAJECTORY_PATH / f'true_y_{seed}.npy', y_true)
    if heading_true is not None:
        np.save(TRAJECTORY_PATH / f'true_heading_{seed}.npy', heading_true)
    if fly_roll is not None:
        np.save(TRAJECTORY_PATH / f'fly_roll_{seed}.npy', fly_roll)