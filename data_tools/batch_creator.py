import logging
import os
import random
import sys

import deepdish as dd
import gc

import numpy as np
import pandas as pd

from brainpy.eeg import EEG
from funcy import merge
from sklearn.cross_validation import train_test_split

import settings
from .utils import OneHotEncoder
from matlab_data_reader import matlab_data_reader, matlab_concat_labels
from data_saver import DataSaver


class BatchCreator(object):
    def __init__(self, batch_size, outdir, avg_group_size=None, eeg_derivation='potential', test_proportion=0.15,
                 seed=42):
        assert eeg_derivation in ['potential', 'electric_field', 'laplacian'], \
            "Derivation '%s' is not supported" % eeg_derivation
        self._batch_size = batch_size
        self._test_proportion = test_proportion
        self._train_index_list, self._test_index_list = train_test_split(
            [(x['filename'], y) for x in settings.FILE_DESCRIPTION for y in range(x['n_trials'])],
            test_size=test_proportion, random_state=seed)
        self._n_trials_train = len(self._train_index_list)
        self._n_trials_test = len(self._test_index_list)
        assert self._batch_size <= self._n_trials_train, "Batch size must be less or equal the number of trials"
        self._index_in_cycle = 0
        self._files_train = []
        self._files_test = []
        self._outdir = outdir
        self._seed = seed
        self._eeg_doc = dict()
        self._avg_group_size = avg_group_size
        self._eeg_derivation = eeg_derivation.lower()
        self._classes = set()
        random.seed(seed)
        random.shuffle(self._train_index_list)

    def _get_batch(self):
        start = self._index_in_cycle
        self._index_in_cycle += self._batch_size
        if self._index_in_cycle > self._n_trials_train:
            random.shuffle(self._train_index_list)
            start = 0
            self._index_in_cycle = self._batch_size
        end = self._index_in_cycle
        return self._train_index_list[start:end]

    def create(self, max_iter):
        logging.info("Creating training and test files to train models")
        self._files_train = []
        self._files_test = []
        batch_info_list = []
        for n, _ in enumerate(range(0, max_iter, self._batch_size)):
            self._files_train.append(os.path.join(self._outdir, "train_%s.hd5" % n))
            batch_info_list.append((self._files_train[-1], self._get_batch()))
        file_exist = dict()
        label_encoder = OneHotEncoder().fit(matlab_concat_labels(d['filename'] for d in settings.FILE_DESCRIPTION))
        self._classes = list(label_encoder.levels)
        for d in settings.FILE_DESCRIPTION:
            logging.info("Processing the EEG file %s", d['filename'])
            eeg = EEG(data_reader=matlab_data_reader).read(d['filename'])
            self._eeg_doc = {"trial_size": int(eeg.trial_size), "group_size": int(self._avg_group_size or 1),
                             "n_comps": {"electric_field": 3}.get(self._eeg_derivation, 1),
                             "n_channels": int(eeg.n_channels), "derivation": str(self._eeg_derivation),
                             "label_encoder": label_encoder.to_json()}
            if (self._avg_group_size or 1) > 1:
                logging.info("Averaging trials")
                eeg.average_trials(self._avg_group_size, inplace=True)
            if self._eeg_derivation != 'potential':
                logging.info("Building the %s derivation", self._eeg_derivation)
                if self._eeg_derivation == "electric_field":
                    eeg.get_electric_field(inplace=True)
                elif self._eeg_derivation == 'laplacian':
                    eeg.get_laplacian(inplace=True)
                else:
                    raise KeyError("Derivation '%s' is not supported", self._eeg_derivation)
            eeg = type('EEG', (), {'data': eeg.data, 'labels': eeg.trial_labels})()
            if eeg.data.ndim == 3:
                eeg.data = eeg.data[:, :, :, np.newaxis]
            eeg.data = eeg.data.reshape(d['n_channels'], d['trial_size'], -1, 3).transpose((2, 0, 1, 3))
            logging.info("Creating batch files")
            for (batch_path, batch_index) in batch_info_list:
                batch2trials = [x[1] for x in batch_index if x[0] == d['filename']]
                if not batch2trials:
                    continue
                if file_exist.get(batch_path):
                    rec = dd.io.load(batch_path)
                    samples = rec['samples']
                    labels = rec['labels']
                else:
                    samples = np.zeros((self._batch_size, eeg.data.shape[1], eeg.data.shape[2], eeg.data.shape[3]))
                    labels = []
                for i, j in enumerate(batch2trials):
                    samples[i, :, :, :] = eeg.data[j, :, :, :]
                    labels.append(list(label_encoder.transform(eeg.labels[j])))
                try:
                    dd.io.save(batch_path, {'samples': samples, 'labels': labels})
                    if not file_exist.get(batch_path):
                        logging.info("Successfully created the training file %s", batch_path)
                    file_exist[batch_path] = True
                except Exception as e:
                    logging.error("Failed to create the training file %s: %s", batch_path, e)
                    sys.exit(1)
            self._files_test.append(os.path.join(self._outdir, "test_%s.hd5" % d['subject']))
            try:
                idx = [j for eeg_file, j in self._test_index_list if eeg_file == d['filename']]
                dd.io.save(self._files_test[-1], {'samples': eeg.data[idx, :, :, :],
                                                  'labels': [list(label_encoder.transform(x)) for x in eeg.labels[idx]]})
                logging.error("Successfully created the test file %s", self._files_test[-1])
            except Exception as e:
                logging.error("Failed to create the test file %s: %s", self._files_test[-1], e)
                sys.exit(1)
            gc.collect()
        logging.info("Finished to create batch files")
        doc = pd.Series(merge(self._eeg_doc,
                              dict(batch_size=int(self._batch_size), n_trials_train=int(self._n_trials_train),
                                   n_trials_test=int(self._n_trials_test), test_proportion=float(self._test_proportion),
                                   index_in_cycle=int(self._index_in_cycle), files_train=list(self._files_train),
                                   files_test=list(self._files_test), outdir=str(self._outdir),
                                   classes=list(self._classes), n_train_batches=len(self._files_train),
                                   n_test_batches=len(self._files_test), n_class=len(self._classes)))).to_dict()
        data_saver = DataSaver()
        try:
            doc_id = data_saver.save(settings.MONGO_DNN_COLLECTION, doc=doc)
            logging.info("Successfully created the new document %s in the DB", doc_id)
        except Exception as e:
            logging.error("Failed to create a new document in the DB: %s", e)
        return self
