import numpy as np

# various functions adapted from the exercise session to compute proprioceptive variables 

def absolute_to_relative_pos(
        pos: np.ndarray, base_pos: np.ndarray, heading: np.ndarray
    ) -> np.ndarray:
    """
    This function converts an absolute position to a relative position
    with respect to a base position and heading of the fly.

    It will be used to obtain the flycentric end effector (leg tip) positions.
    """

    rel_pos = pos - base_pos
    heading = heading / np.linalg.norm(heading)
    angle = np.arctan2(heading[1], heading[0])
    rot_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    pos_rotated = np.dot(rel_pos, rot_matrix.T)
    return pos_rotated


def get_stride_length(fly_pos, heading, end_effector_pos, last_end_effector_pos = None):
    """
    This function calculates the stride length of the fly by calculating the difference in the end effector position
    of the fly between two consecutive time steps. FOR ALL STEPS

    In this function the end_effector_pos is converted to a flycentric coordinate system using the absolute_to_relative_pos
    function. The stride length is calculated as the difference in the end effector position between two consecutive time steps
    in the flycentric coordinate system.
    """
    stride_length = []

    # TODO adapt it to compute as it goes, more natural
    # WE DONT HAVE FLY_POS so have to do it within a given window_vision or for each step
    # we would like to have it to switch to a fly centric vision
    for i in range(len(fly_pos)):
        rel_pos = absolute_to_relative_pos(end_effector_pos[i], fly_pos[i], heading[i])
        if last_end_effector_pos is None:
            ee_diff = np.zeros_like(rel_pos)
        else:
            ee_diff = rel_pos - last_end_effector_pos
        last_end_effector_pos = rel_pos
        
        stride_length.append(ee_diff)

    return np.array(stride_length)

def get_stride_length_instantaneous(fly_pos, heading, end_effector_pos, last_end_effector_pos = None):
    """
    This function calculates the stride length of the fly by calculating the difference in the end effector position
    of the fly between two consecutive time steps.

    In this function the end_effector_pos is converted to a flycentric coordinate system using the absolute_to_relative_pos
    function. The stride length is calculated as the difference in the end effector position between two consecutive time steps
    in the flycentric coordinate system.
    """

    # WE DONT HAVE FLY_POS so have to do it within a given window_vision or for each step
    # we would like to have it to switch to a fly centric vision
    
    rel_pos = absolute_to_relative_pos(end_effector_pos, fly_pos, heading)
    if last_end_effector_pos is None:
        ee_diff = np.zeros_like(rel_pos)
    else:
        ee_diff = rel_pos - last_end_effector_pos

    return ee_diff

def extract_proprioceptive_variables_from_steps(stride_length, contact_force, time_scale = 0.12, contact_force_thr = 1, dt = 1e-4):
    """
    This function calculates the proprioceptive heading and distance signals from step information.

    The proprioceptive HEADING signal is calculated as the difference in the stride length
    between the left and right side of the fly.

    The proprioceptive DISTANCE signal is calculated as the sum of the stride length
    of the left and right side of the fly.

    time_scale = 0.12 from the exercises
    dt = 1e-4 from the simulation timestep TODO check
    contact_force_thr = 1
    """
    window_len = int(time_scale / dt)

    # Calculate total stride (Σstride) for each side
    # Extract the stride length for each side along the x axis
    stride_left = stride_length[:, :3 , 0] # first 3 legs
    stride_right = stride_length[:, 3:, 0] # last 3 legs

    contact_mask = contact_force > contact_force_thr  # (L, 6)
    
    # Calculate the stride length for the stance period (touching the ground)
    stride_left = (stride_left * contact_mask[:, :3]).sum(axis=1)
    stride_right = (stride_right * contact_mask[:, 3:]).sum(axis=1)

    stride_total_left = np.cumsum(stride_left, axis=0)
    stride_total_right = np.cumsum(stride_right, axis=0)

    # Calculate difference in Σstride over proprioceptive time window (ΔΣstride)
    stride_total_diff_left = stride_total_left[window_len:] - stride_total_left[:-window_len]
    stride_total_diff_right = stride_total_right[window_len:] - stride_total_right[:-window_len]
    
    # Calculate sum and difference in ΔΣstride over two sides (final proprioceptive signals)
    proprioceptive_distance_pred = stride_total_diff_left + stride_total_diff_right
    proprioceptive_heading_pred = stride_total_diff_left - stride_total_diff_right

    return proprioceptive_heading_pred, proprioceptive_distance_pred


def compute_proprioceptive_variables(obs_buffer): # TODO put it to self later
    """
    This function computes the proprioceptive variables for the fly.
    It uses the stride length and contact forces to calculate the proprioceptive heading and distance signals.
    """
    # Assuming obs_buffer is a list of observations with keys "fly_pos", "heading", "end_effectors", "contact_forces"
    end_efector_pos = np.array([obs['end_effectors'] for obs in obs_buffer])
    heading = np.array([obs['heading'] for obs in obs_buffer])
    contact_forces = np.array([obs['contact_forces'] for obs in obs_buffer])

    AAA = obs["fly_pos"][:, :2] # TODO FLY POSITION
    stride_length = get_stride_length(AAA, heading, end_efector_pos)

    stride_length_difference, stride_length_sum = extract_proprioceptive_variables_from_steps(stride_length, contact_forces)

    return stride_length_difference, stride_length_sum
