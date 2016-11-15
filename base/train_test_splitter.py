import gc
import numpy as np
from brainpy.eeg import EEG

from base.base_data import BaseData
from base.eeg_dataset import EEGDataset


class TrainTestSplitter(BaseData):
    def __init__(self, test_proportion=0.2, random_seed=42, channel_scheme='single'):
        super(TrainTestSplitter, self).__init__()
        assert 0 < test_proportion < 1, "%s: test proportion must be a float number between 0 and 1 got %s" \
                                        % (self.__class__.__name__, test_proportion)
        self.test_proportion = test_proportion
        self.random_seed = random_seed
        self.datasets = []
        self._channel_scheme = channel_scheme
        self._params = {'test_proportion': self.test_proportion, 'random_seed': self.random_seed,
                        'channel_scheme': self._channel_scheme}

    @property
    def channel_scheme(self):
        return self._channel_scheme

    @property
    def params(self):
        return self._params

    def eval(self, eeg):
        assert isinstance(eeg, EEG)
        np.random.rand(self.random_seed)
        j = np.random.rand(len(eeg.trial_labels)).argsort()
        test_size = int(self.test_proportion * eeg.n_trials)
        train_labels = eeg.trial_labels[test_size:]
        test_labels = eeg.trial_labels[:test_size]
        self.datasets = []
        if self._channel_scheme == 'single':
            eeg = eeg.data.reshape((eeg.n_channels, eeg.trial_size, eeg.n_trials, -1)).transpose((2, 1, 3, 0))
            for n in range(eeg.shape[-1]):
                gc.collect()
                ds = EEGDataset("channel_%s" % n, eeg[j[test_size:], :, :, n].squeeze(), train_labels,
                                eeg[j[:test_size], :, :, n].squeeze(), test_labels)
                self.datasets.append(ds)
        else:
            eeg = eeg.data.transpose((1, 0, 2)).squeeze()
            ds = EEGDataset("multichannel", eeg[j[test_size:], :, :], train_labels, eeg[j[test_size:], :, :],
                            test_labels)
            self.datasets = [ds]
        return self
