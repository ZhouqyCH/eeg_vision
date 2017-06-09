import glob
import os
from collections import defaultdict

import deepdish as dd
import numpy as np


class BootstrapBatch(object):
    def __init__(self, arr, labels, group_size_max, batch_size, seed=42, auto_remove_files=True):
        self._seed = seed
        self.arr = arr
        self.labels = labels
        self._indices = range(len(labels))
        self._batch_size = batch_size
        self.group_size_max = group_size_max
        self._curr_group_size = 1
        self._curr_file_index = None
        self._cache = defaultdict(list)
        self._auto_remove_files = auto_remove_files
        np.random.rand(seed)

    def _add_to_group(self, key, arr):
        self._cache[key].append(arr)
        group = None
        if len(self._cache[key]) >= self._curr_group_size:
            values_list = self._cache.pop(key)
            group = sum(values_list) / len(values_list)
        return group, key

    def next_batch(self):
        batch = []
        self._curr_group_size = np.random.choice(range(1, self.group_size_max + 1))
        while len(batch) < self._batch_size:
            k = np.random.choice(self._indices)
            group, label = self._add_to_group(self.labels[k], self.arr[k])
            if group is not None:
                batch.append((group, label))
        return batch

    def create(self, max_iter, path, prefix):
        bm = BootstrapBatchFiles(auto_remove=self._auto_remove_files)
        for i in range(max_iter):
            batch = self.next_batch()
            samples = np.asarray(map(lambda x: x[0], batch))
            labels = map(lambda x: x[1], batch)
            batch_file = os.path.join(path, "%s%s.hd5" % (prefix, i+1))
            dd.io.save(batch_file, {'samples': samples,
                                    'labels': labels})
            bm.append(batch_file)
        return bm

    def load(self, path, prefix):
        bm = BootstrapBatchFiles(auto_remove=self._auto_remove_files, batch_size=self._batch_size)
        pattern = os.path.join(path, prefix + "*.hd5")
        for batch_file in glob.glob(pattern):
            bm.append(batch_file)
        return bm


class BootstrapBatchFiles(object):
    def __init__(self, batch_files=None, auto_remove=True, batch_size=None):
        self._curr_file_index = 0
        self._batch_files = batch_files or []
        self._auto_remove = auto_remove
        self.size = batch_size

    @property
    def count(self):
        return self._curr_file_index

    @property
    def count_max(self):
        return len(self._batch_files)

    def next_batch(self):
        if self._curr_file_index >= self.count_max:
            return None, None
        path = self._batch_files[self._curr_file_index]
        self._curr_file_index += 1
        data = dd.io.load(path)
        if self._auto_remove:
            os.remove(path)
        return data['samples'], data['labels']

    def append(self, filename):
        self._batch_files.append(filename)
        return self

    def remove_batch_files(self):
        for file_name in self._batch_files:
            if os.path.isfile(file_name):
                os.remove(file_name)
        return self
