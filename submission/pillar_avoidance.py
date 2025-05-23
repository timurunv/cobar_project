import numpy as np
from tqdm import trange

def calc_ipsilateral_speed(deviation, is_found):
    if not is_found:
        return 1.0
    else:
        return np.clip(deviation * 2 - 1, 0.2, 1.2)
    
def calc_ipsilateral_speed_vision(proportion, is_found):
    return np.clip(proportion * 0.7 / 0.01, 0.4, 0.7)


def compute_pillar_avoidance(visual_features):
    left_deviation = 1 - visual_features[1]
    right_deviation = visual_features[4]
    left_found = visual_features[2] > 0.01 #if area is bigger than 1% of the ommatidia
    right_found = visual_features[5] > 0.01
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