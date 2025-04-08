import numpy as np

def compute_olfaction_control_signal(obs, attractive_gain = -500, aversive_gain = 80):
    
    attractive_intensities = np.average(obs["odor_intensity"][0, :].reshape(2, 2), axis=0, weights=[9, 1])
    aversive_intensities = np.average(obs["odor_intensity"][1, :].reshape(2, 2), axis=0, weights=[10, 0])
    
    attractive_bias = (
        attractive_gain
        * (attractive_intensities[0] - attractive_intensities[1])
        / np.maximum(attractive_intensities.mean(), 1e-6)
    )
    aversive_bias = (
        aversive_gain
        * (aversive_intensities[0] - aversive_intensities[1])
        / np.maximum(aversive_intensities.mean(), 1e-6)
    )
    effective_bias = aversive_bias + attractive_bias
    effective_bias_norm = np.tanh(effective_bias**2) * np.sign(effective_bias)
    assert np.sign(effective_bias_norm) == np.sign(effective_bias)

    control_signal = np.zeros((2,))
    side_to_modulate = int(effective_bias_norm > 0)
    modulation_amount = np.abs(effective_bias_norm) * 0.8
    control_signal[side_to_modulate] -= modulation_amount

    return control_signal