import numpy as np


def get_eeg_shape(data, trial_size):
    data = data.squeeze()
    assert data.ndim in [2, 3]
    n_channels = data.shape[0]
    n_trials = data.shape[1] // trial_size
    n_comps = data.shape[2] if data.ndim == 3 else 1
    assert n_trials * trial_size == data.shape[1]
    return n_channels, n_trials, n_comps


def eeg_reshape(data, trial_size, single_channel):
    n_chans, n_trials, n_comps = get_eeg_shape(data, trial_size)
    if n_comps == 1:
        data = np.array([data[:, i:(i + trial_size)].ravel() for i in range(0, trial_size * n_trials, trial_size)])
        if single_channel:
            data = data.reshape((n_trials, trial_size, n_chans))
    else:
        data = np.array([data[:, i:(i + trial_size), :].ravel() for i in range(0, trial_size * n_trials, trial_size)])
        if single_channel:
            data = data.reshape((n_trials, trial_size * n_comps, n_chans))
    return data
