from pathlib import Path
import numpy as np
import os
BASE_DIR = Path.cwd()
TEST_PATH = BASE_DIR / 'outputs' /'test_heading'

TEST_PATH.mkdir(parents=True, exist_ok=True)

def test_heading(counter, obs, obs_vision, fly_roll_hist, estimated_orient_change, debug=False):
    # see jupyter notebook out_tests/test_heading-ipynb for the spot where this is use
    if counter % 500 == 0: # every 100 steps
        np.save(TEST_PATH / f'vision_{counter}.npy', obs_vision)
    if obs.get('reached_odour', False):
        cum_orientations_final = np.cumsum(estimated_orient_change)
        np.save(TEST_PATH / 'cum_orientations_final.npy', cum_orientations_final)
        if debug: 
            fly_roll_hist = np.unwrap(fly_roll_hist, discont=np.pi) # if debug mode
        np.save(TEST_PATH / 'fly_roll_hist.npy', fly_roll_hist)
