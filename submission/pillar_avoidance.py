import numpy as np
from tqdm import trange

def calc_ipsilateral_speed(deviation, is_found):
    if not is_found:
        return 1.0
    else:
        return np.clip(1 - deviation * 5, 0.4, 1.2) #TODO can tune and maybe baisser a 5


def compute_pillar_avoidance(visual_features, obs_contact_forces):
    left_deviation = 1 - visual_features[1] #1 for x axis of one side ok
    right_deviation = visual_features[4] #4 for x axis of one side ok
    left_found = visual_features[2] > 0.001 #if area is bigger than 1% of the ommatidia
    right_found = visual_features[5] > 0.001
    print(visual_features[2], "left eye percentage", visual_features[5], "right eye percentage", obs_contact_forces, "obs velocity de merde de gon")
    if not left_found:
        left_deviation = np.nan
    if not right_found:
        right_deviation = np.nan
    object_detected = left_found or right_found
    control_signal = np.array(
        [
            calc_ipsilateral_speed(right_deviation, right_found),
            calc_ipsilateral_speed(left_deviation, left_found),
        ]
    )
    return control_signal, object_detected