import os

import numpy as np
import pandas as pd
from scipy.io import loadmat


def matlab_data_reader(file_name, labels='categoryLabels'):
    path = os.path.split(file_name)[0]
    elect_file = os.path.join(path, "elect.csv")
    electrodes = pd.read_csv(elect_file, index_col=False).to_dict(orient="records")
    mat = loadmat(file_name)
    trial_size = mat['N'].ravel()[0]
    data = mat.pop('X')
    n_trials, n = data.shape
    n_channels = n // trial_size
    assert n == n_channels * trial_size
    data = np.array([data[:, k * trial_size:][:, :trial_size].ravel() for k in range(n_channels)])
    assert data.shape == (n_channels, trial_size * n_trials)
    return dict(sampling_rate=mat['Fs'].ravel()[0],
                data=data[:, :, np.newaxis],
                electrodes=electrodes,
                trial_size=trial_size,
                subject=mat['sub'].ravel()[0],
                trial_labels=mat[labels].ravel(),
                der_code=0,
                group_size=1)


cached_data = dict()


def get_matlab_labels(filename, labels='categoryLabels'):
    data = cached_data.get(filename)
    if not data:
        data = loadmat(filename)[labels].ravel().tolist()
        cached_data[filename] = data
    return data
