import numpy as np
from .utils import compute_optic_flow, prepare_fly_vision
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import os

# various functions adapted from the exercise session to compute proprioceptive variables 

def predict_roll_change(vision_buffer:list, n_top_pixels=8):
    post_img = prepare_fly_vision(vision_buffer[-1], n_top_pixels=n_top_pixels)
    pre_img = prepare_fly_vision(vision_buffer[0], n_top_pixels=n_top_pixels)
    
    flow = compute_optic_flow(pre_img, post_img)

    # calculate the mean optic flow vector in the x direction
    mean_x_flow = np.mean(flow[..., 0])
    return mean_x_flow


def get_stride_length(end_effector_pos, heading, last_end_effector_pos = None):
    """
    This function calculates the stride length of the fly by computing the difference in the end effector 
    position between two consecutive time steps in the flycentric coordinate system.

    args:
        end_effector_pos: (6, 2) - position of the end effectors in world coordinates
        last_end_effector_pos: (6, 2) - position of the end effectors in world coordinates at the previous time step
    """
    if last_end_effector_pos is None:
        ee_diff = np.zeros_like(end_effector_pos)
    else:
        #project on the heading direction - WRONG
        # rot_matrix = np.array([[np.cos(-heading), -np.sin(-heading)], [np.sin(-heading), np.cos(-heading)]])
        # end_effector_pos_rotated = np.dot(end_effector_pos, rot_matrix.T)
        ee_diff = end_effector_pos - last_end_effector_pos
    last_end_effector_pos = end_effector_pos
    return ee_diff, last_end_effector_pos

def extract_proprioceptive_variables_from_stride(stride_length: np.array, contact_force : np.array, window_len : int, contact_force_thr: float = 1):
    """
    This function calculates the proprioceptive heading and distance signals from step information.

    The proprioceptive HEADING signal is calculated as the difference in the stride length
    between the left and right side of the fly.

    The proprioceptive DISTANCE signal is calculated as the sum of the stride length
    of the left and right side of the fly.
    contact_force_thr = 1
    args:
        stride_length: (2*window_len, 6, 2) - position of the end effectors in 
        contact_force: (2*window_len, 6, 3) - contact forces of the end effectors 
        contact_force_thr: float - threshold for the contact force to be considered as a contact
    """
    # Calculate total stride (Σstride) for each side
    # Extract the stride length for each side along the x axis
    stride_left = stride_length[:, :3 , 0] # first 3 legs
    stride_right = stride_length[:, 3:, 0] # last 3 legs
    
    # take force only on z axis
    contact_mask = contact_force[:,:,-1] > contact_force_thr  # (window_len, 6) 

    # Calculate the stride length for the stance period (touching the ground)
    stride_left = (stride_left * contact_mask[:, :3]).sum(axis=1)
    stride_right = (stride_right * contact_mask[:, 3:]).sum(axis=1)

    stride_total_left = np.cumsum(stride_left, axis=0)
    stride_total_right = np.cumsum(stride_right, axis=0)

    # Calculate difference in Σstride over proprioceptive time window (ΔΣstride) (window_len,)
    stride_total_diff_left = stride_total_left[window_len:] - stride_total_left[:-window_len]
    stride_total_diff_right = stride_total_right[window_len:] - stride_total_right[:-window_len]

    # Calculate sum and difference in ΔΣstride over two sides (final proprioceptive signals)
    proprioceptive_distance_pred = stride_total_diff_left + stride_total_diff_right
    proprioceptive_heading_pred = stride_total_diff_left - stride_total_diff_right
    
    return proprioceptive_heading_pred, proprioceptive_distance_pred


def load_proprioceptive_models():
    path_proprio = os.path.join(os.getcwd(), 'submission', 'proprioceptive_models.json')
    with open(path_proprio, 'r') as f:
        models_data = json.load(f)

    # Reconstruct poly1d models
    prop_heading_model = np.poly1d([models_data['prop_heading_model']['slope'],models_data['prop_heading_model']['intercept']])
    prop_disp_model = np.poly1d([models_data['prop_disp_model']['slope'],models_data['prop_disp_model']['intercept']])

    # Optic flow heading prediction model
    path_heading_flow = os.path.join(os.getcwd(), 'submission', 'heading_optic_models.json')
    with open(path_heading_flow, 'r') as f:
        heading_models_data = json.load(f)
    
    heading_flow_model_univariate = LinearRegression()
    heading_flow_model_univariate.coef_ = np.array([[heading_models_data['heading_flow_model']['slope']]])
    heading_flow_model_univariate.intercept_ = np.array([heading_models_data['heading_flow_model']['intercept']])
    heading_flow_model_univariate._residues = None;   heading_flow_model_univariate.n_features_in_ = 1

    # Multivariate model
    # heading_flow_model.coef_ = np.array([[heading_models_data['heading_flow_model']['slopes']]])
    # heading_flow_model.intercept_ = np.array([heading_models_data['heading_flow_model']['intercept']])
    # heading_flow_model._residues = None  # Legacy, not needed usually
    # heading_flow_model_multivar.n_features_in_ = 2


    # velocity model
    path_velocity = os.path.join(os.getcwd(), 'submission', 'velocity_model.json')
    with open(path_velocity, 'r') as f:
        velocity_models_data = json.load(f)
    velocity_model = LinearRegression()
    velocity_model.coef_ = np.array([[velocity_models_data['coefs']]])
    velocity_model.intercept_ = np.array([velocity_models_data['intercept']])
    velocity_model._residues = None;   velocity_model.n_features_in_ = 1

    return prop_heading_model, prop_disp_model, heading_flow_model_univariate, velocity_model


def absolute_to_relative_pos( # NOT USED
        pos: np.ndarray, base_pos: np.ndarray, heading: np.ndarray
    ) -> np.ndarray:
    """
    This function converts an absolute position to a relative position
    with respect to a base position and heading of the fly.

    It will be used to obtain the flycentric end-effector (leg tip) positions.
    """

    rel_pos = pos - base_pos
    heading = heading / np.linalg.norm(heading)
    angle = np.arctan2(heading[1], heading[0])
    rot_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    pos_rotated = np.dot(rel_pos, rot_matrix.T)
    return pos_rotated



def get_stride_lengths(end_effector_pos): # Not used since done step by step
    """
    This function calculates the stride length of the fly by calculating the difference in the end effector position
    of the fly between two consecutive proprioceptive time steps. 

    In this function the end_effector_pos is already in flycentric coordinates.
    The stride length is calculated as the difference in the end effector position between two consecutive time steps
    in the flycentric coordinate system.
    """
    last_end_effector_pos = None
    stride_length = []
    
    for i in range(end_effector_pos.shape[0]):
        pos = end_effector_pos[i,:,:]
        if last_end_effector_pos is None:
            ee_diff = np.zeros_like(pos)
        else:
            ee_diff = pos - last_end_effector_pos
        last_end_effector_pos = pos
        
        stride_length.append(ee_diff)

    return np.array(stride_length)