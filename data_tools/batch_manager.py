import deepdish as dd
import numpy as np

import settings
from .data_loader import DataLoader


class BatchManager(object):
    def __init__(self):
        self._curr_batch = 0
        self._curr_test = 0
        self._curr_test_subj = None
        self._batch_id = None
        self._doc = dict()
        self._data = dict()
        self._coll = settings.MONGO_DNN_COLLECTION

    def load(self, db_id):
        self._doc = DataLoader.load(settings.MONGO_DNN_COLLECTION, _id=db_id)
        if self._doc:
            self._doc = self._doc[0]
        return self

    def samples(self, typ):
        return self._data[typ]['samples']

    def labels(self, typ):
        return self._data[typ]['labels']

    def __getattr__(self, item):
        if item not in self._doc:
            raise AttributeError("'%s' is not a valid attribute", item)
        return self._doc[item]

    def _next(self, typ, curr, files):
        if curr >= len(files):
            return False
        data = dd.io.load(files[curr])
        self._data[typ] = {'samples': data['samples'], 'labels': data['labels']}
        return True

    def next_batch(self):
        state = self._next('train', self._curr_batch, self.files_train)
        self._curr_batch += int(state)
        return state

    def next_test(self):
        state = self._next('test', self._curr_test, self.files_test)
        self._curr_test += int(state)
        return state

    def aggregate_tests(self):
        samples = None
        labels = []
        while self.next_test():
            if samples is None:
                samples = self.samples('test')
            else:
                samples = np.r_[samples, self.samples('test')]
            labels += self.labels('test')
            assert len(labels) == samples.shape[0], \
                "Oops! Something went wrong! Mismatch between number of samples and labels"
        self._data['test'] = {'samples': samples, 'labels': labels}
        return self
