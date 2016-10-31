import numpy as np
from brainpy.eeg import EEG

from base.base_data import BaseData


class TrainTestSplitter(BaseData):
    def __init__(self, test_proportion=0.2, random_seed=42, channel_scheme='single'):
        super(TrainTestSplitter, self).__init__()
        assert 0 < test_proportion < 1, "%s: test proportion must be a float number between 0 and 1 got %s" \
                                        % (self.__class__.__name__, test_proportion)
        self.test_proportion = test_proportion
        self.random_seed = random_seed
        self.train = np.array([])
        self.train_labels = np.array([])
        self.test = np.array([])
        self.test_labels = np.array([])
        self._channel_scheme = channel_scheme
        self._params = {'test_proportion': self.test_proportion, 'random_seed': self.random_seed,
                        'channel_scheme': self._channel_scheme}

    @property
    def mango_collection(self):
        return "class_splitter"

    @property
    def channel_scheme(self):
        return self._channel_scheme

    def get_params(self):
        return self._params.copy()

    def eval(self, eeg):
        assert isinstance(eeg, EEG)
        n = len(eeg.trial_labels)
        np.random.rand(self.random_seed)
        j = np.random.rand(n).argsort()
        test_size = int(self.test_proportion * eeg.n_trials)
        if self._channel_scheme == 'single':
            self.train = eeg.data.reshape((eeg.n_channels, eeg.trial_size, eeg.n_trials, -1))\
                .transpose((2, 1, 0, 3)).squeeze()
        else:
            self.train = eeg.data.reshape((eeg.n_channels, eeg.trial_size, eeg.n_trials, -1))\
                .transpose((2, 0, 1, 3)).squeeze()
        self.test = self.train[j[:test_size], :, :]
        self.test_labels = eeg.trial_labels[:test_size]
        self.train = self.train[j[test_size:], :, :]
        self.train_labels = eeg.trial_labels[test_size:]
        self._params.update(eeg.get_params())
        return self
