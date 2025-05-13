import numpy as np

def is_ball(rgb_features, ball_threshold):
#Check if red object is close-by
        red_channel = rgb_features[:, :, :, 0]
        green_channel = rgb_features[:, :, :, 1]
        blue_channel = rgb_features[:, :, :, 2]

        red_thresh = (red_channel > green_channel) & (red_channel > blue_channel)
        nb_red = red_thresh.sum(axis=(1,2)) # shape (2, 721)
        nb_pix = rgb_features.shape[1] * rgb_features.shape[2]
        red_proportion = nb_red / nb_pix # shape (2, )
        if red_proportion.max() > ball_threshold: # if red proportion is bigger than threshold
            ball_alert = True
        else:
            ball_alert = False

        return ball_alert