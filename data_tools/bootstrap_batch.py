import os
from collections import defaultdict

import deepdish as dd
import logging
import numpy as np
import pandas as pd
from brainpy.eeg import EEG
from funcy.colls import merge
from sklearn.cross_validation import train_test_split

import settings
from data_tools import matlab_data_reader
from data_tools.data_saver import DataSaver
from data_tools.utils import OneHotEncoder


class BaseBootstrapBatch(object):
    def __init__(self, arr, labels, group_size, batch_size, seed=42):
        self._seed = seed
        self.arr = arr
        self.labels = labels
        self._indices = range(len(labels))
        self._batch_size = batch_size
        self.group_size = group_size
        self._cache = defaultdict(list)
        np.random.rand(seed)

    def _add(self, key, arr):
        self._cache[key].append(arr)
        if len(self._cache[key]) >= self.group_size:
            values_list = self._cache.pop(key)
            m = sum(values_list) / len(values_list)
            return self.transform_key(key), m
        return None, None

    def transform_key(self, key):
        return key

    def next_batch(self):
        batch = []
        while len(batch) < self._batch_size:
            k = np.random.choice(self._indices)
            lab, mean = self._add(self.labels[k], self.arr[k])
            if mean is not None:
                batch.append((lab, mean))
        return batch


class EEGReader(object):
    def __init__(self, subject, derivation):
        self.info = {'subject': subject, 'derivation': derivation}
        self.eeg = None

    def load_eeg(self):
        d = settings.SUBJECT_DICT[self.info['subject']]
        self.eeg = EEG(data_reader=matlab_data_reader).read(d['filename'])
        self.info['labels'] = self.eeg.trial_labels
        self.info['classes'] = list(set(self.info['labels']))
        self.info['n_classes'] = len(self.info['classes'])
        self.info = merge(self.info, d)
        if self.info['derivation'] == 'potential':
            self.info['n_comps'] = 1
        elif self.info['derivation'] == 'laplacian':
            self.eeg.get_laplacian(inplace=True)
            self.info['n_comps'] = 1
        elif self.info['derivation'] == "electric_field":
            self.eeg.get_electric_field(inplace=True)
            self.info['n_comps'] = 3
        else:
            raise KeyError("Derivation '%s' is not supported", self.info['derivation'])
        return self


class BatchCreator(BaseBootstrapBatch):
    def __init__(self, subject, batch_size, group_size, derivation, test_proportion, out_dir, seed=42):
        logging.info("%s: Loading data from subject %s", self, subject)
        eeg = EEGReader(subject, derivation).load_eeg()
        logging.info("%s: Successfully loaded data", self)
        self.info = merge(eeg.info, {'batch_size': batch_size, 'test_proportion': test_proportion, 'out_dir': out_dir,
                                     'seed': seed})
        eeg = eeg.eeg.data
        labels = self.info['labels']
        self.label_encoder = OneHotEncoder().fit(labels)
        train, self.test, train_labels, self.test_labels = train_test_split(eeg, labels, test_size=test_proportion,
                                                                            random_state=seed)
        super(BatchCreator, self).__init__(train, train_labels, group_size, batch_size, seed=seed)

    def __str__(self):
        return self.__class__.__name__

    def transform_key(self, key):
        return self.label_encoder.transform(key)

    def create(self, max_iter):
        logging.info("%s: Creating the batch files", self)
        for i in range(max_iter):
            batch_file = os.path.join(self.info['out_dir'], "%s_train_%s.hd5" % (self.info['subject'], i+1))
            batch = self.next_batch()
            samples = np.concatenate(map(lambda x: x[1], batch))
            labels = map(lambda x: x[0], batch)
            dd.io.save(batch_file, {'samples': samples, 'labels': labels})
            logging.info("%s: Successfully exported train data to %s", self, batch_file)
        test_file = os.path.join(self.info['out_dir'], "%s_test.hd5" % self.info['subject'])
        dd.io.save(test_file, {'samples': self.test,
                               'labels': [list(self.label_encoder.transform(x)) for x in self.test_labels]})
        logging.error("%s: Successfully created the test file %s", self, test_file)
        self.info['n_train_batches'] = max_iter
        self.info['n_test_batches'] = 1
        self.info['files_test'] = [test_file]
        logging.info("Finished to create batch files")
        doc = pd.Series(self.info).to_dict()
        data_saver = DataSaver()
        try:
            doc_id = data_saver.save(settings.MONGO_DNN_COLLECTION, doc=doc)
            logging.info("Successfully created the new document %s in the DB", doc_id)
        except Exception as e:
            logging.error("Failed to create a new document in the DB: %s", e)
        return self
