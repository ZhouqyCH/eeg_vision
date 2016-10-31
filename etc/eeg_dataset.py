import numpy as np
from brainpy.eeg import EEG
from sklearn.cross_validation import train_test_split

from etc.data_reader import data_reader


class EEGDataSet(object):
    def __init__(self, data, labels):
        self._eeg = EEG()
        self._cycles_completed = 0
        self._index_in_cycle = 0
        self._data = data
        self._labels = labels
        self._n_class = len(labels[0])

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def n_trials(self):
        return self._data.shape[0]

    @property
    def n_channels(self):
        return self._data.shape[1]

    @property
    def trial_size(self):
        return self._data.shape[2]

    @property
    def n_comps(self):
        if self._data.ndim == 4:
            return self._data.shape[3]
        return 1

    @property
    def n_class(self):
        return self._n_class

    @property
    def cycles_completed(self):
        return self._cycles_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_cycle
        self._index_in_cycle += batch_size
        if self._index_in_cycle > self.n_trials:
            # Finished epoch
            self._cycles_completed += 1
            # Shuffle the data
            perm = np.arange(self.n_trials)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_cycle = batch_size
            assert batch_size <= self.n_trials
        end = self._index_in_cycle
        return self._data[start:end], self._labels[start:end]


def one_hot_encoder(arr):
    levels = np.unique(arr)
    mat = np.eye(len(levels))
    d = {v: mat[:, j] for j, v in enumerate(levels)}
    return np.array([d[v] for v in arr])


# TODO: RESHAPE DATA FOR POTENTIAL AND LAPLACIAN
def build_data_sets(file_name, avg_group_size=None, derivation=None, random_state=42, test_size=0.2):
    eeg = EEG(data_reader=data_reader).read(file_name)
    if avg_group_size:
        eeg.average_trials(avg_group_size, inplace=True)
    if derivation.lower() == "ef":
        eeg.get_electric_field(inplace=True)
        eeg.data = eeg.data.reshape(eeg.n_channels, eeg.trial_size, -1, 3).transpose((2, 0, 1, 3))
    elif derivation.lower() == 'lap':
        eeg.get_laplacian(inplace=True)
    labels = one_hot_encoder(eeg.trial_labels)
    X_train, X_test, y_train, y_test = train_test_split(eeg.data, labels, test_size=test_size,
                                                        random_state=random_state)
    return type('EEGDataSet', (), {'train': EEGDataSet(X_train, y_train), 'test': EEGDataSet(X_test, y_test)})
