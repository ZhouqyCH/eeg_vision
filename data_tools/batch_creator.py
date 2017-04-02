import logging
import os
import random
import sys

import numpy as np
from brainpy.eeg import EEG
from sklearn.cross_validation import train_test_split

import settings
from data_reader import matlab_data_reader
from data_saver import DataSaver


class BatchCreator(object):
    def __init__(self, batch_size, outdir, avg_group_size=None, eeg_derivation='potential', test_proportion=0.15,
                 seed=42):
        self._batch_size = batch_size
        self._test_proportion = test_proportion
        self._index_train, self._index_test = train_test_split(
            [(x['filename'], y) for x in settings.FILE_DESCRIPTION for y in range(x['n_trials'])],
            test_size=test_proportion, random_state=seed)
        self._n_trials_train = len(self._index_train)
        self._n_trials_test = len(self._index_test)
        assert self._batch_size <= self._n_trials_train, "Batch size must be less or equal the number of trials"
        self._index_in_cycle = 0
        self._files_train = []
        self._files_test = []
        self._outdir = outdir
        self._seed = seed
        self._avg_group_size = avg_group_size
        self._eeg_derivation = eeg_derivation.lower()
        random.seed(seed)
        random.shuffle(self._index_train)

    def _get_batch(self):
        start = self._index_in_cycle
        self._index_in_cycle += self._batch_size
        if self._index_in_cycle > self._n_trials_train:
            random.shuffle(self._index_train)
            start = 0
            self._index_in_cycle = self._batch_size
        end = self._index_in_cycle
        return self._index_train[start:end]

    def _create_test(self):
        pass

    def _average_eeg_trials(self, eeg):
        if (self._avg_group_size or 1) > 1:
            eeg.average_trials(self._avg_group_size, inplace=True)
        return eeg

    def _get_eeg_derivation(self, eeg):
        if self._eeg_derivation == "electric_field":
            eeg.get_electric_field(inplace=True)
            eeg.data = eeg.data.reshape(eeg.n_channels, eeg.trial_size, -1, 3).transpose((2, 0, 1, 3))
        elif self._eeg_derivation == 'laplacian':
            eeg.get_laplacian(inplace=True)
        return eeg

    def create(self, max_iter):
        logging.info("Creating batch files for DNN training from available EEG signals")
        self._files_train = []
        self._files_test = []
        batches = []
        for n, _ in enumerate(range(0, max_iter, self._batch_size)):
            self._files_train.append(os.path.join(self._outdir, "batch_%s.npy" % n))
            batches.append((self._files_train[-1], self._get_batch()))
        file_exist = dict()
        for d in settings.FILE_DESCRIPTION:
            logging.info("Processing the EEG file %s", d['filename'])
            eeg = EEG(data_reader=matlab_data_reader).read(d['filename'])
            if (self._avg_group_size or 1) > 1:
                logging.info("Averaging trials")
                eeg = self._average_eeg_trials(eeg)
            if self._eeg_derivation != 'potential':
                logging.info("Building the %s derivation", self._eeg_derivation)
                eeg = self._get_eeg_derivation(eeg)
            logging.info("Creating batch files")
            for (batch_file, batch_index) in batches:
                batch2trials = [(n, x[1]) for n, x in enumerate(batch_index) if x[0] == d['filename']]
                if not batch2trials:
                    continue
                if file_exist.get(batch_file):
                    with open(batch_file, "rb") as fp:
                        arr = np.load(fp)
                else:
                    arr = np.zeros((self._batch_size, eeg.n_channels, eeg.trial_size, eeg.n_comps))
                for (i, j) in batch2trials:
                    trial = eeg.get_single_trial(j)
                    arr[i, :, :, :] = trial
                try:
                    with open(batch_file, "wb") as fp:
                        np.save(fp, arr)
                        if not file_exist.get(batch_file):
                            logging.info("Successfully created the batch file %s", batch_file)
                        file_exist[batch_file] = True
                except Exception as e:
                    logging.error("Failed to create the batch file %s: %s", batch_file, e)
                    sys.exit(1)
            arr = eeg.get_trials([j for eeg_file, j in self._index_test if eeg_file == d['filename']])\
                .transpose((3, 0, 1, 2))
            self._files_test.append(os.path.join(self._outdir, "test_%s.npy" % eeg.subject))
            try:
                with open(self._files_test[-1], "wb") as fp:
                    np.save(fp, arr)
                    logging.error("Successfully created the test file %s", self._files_test[-1])
            except Exception as e:
                logging.error("Failed to create the test file %s: %s", self._files_test[-1], e)
                sys.exit(1)

    def to_db(self):
        doc = dict(batch_size=self._batch_size, n_trials_train=self._n_trials_train, n_trials_test=self._n_trials_test,
                   test_proportion=self._test_proportion, index_in_cycle=self._index_in_cycle,
                   files_train=self._files_train, files_test=self._files_test, outdir=self._outdir,
                   n_batches=len(self._files_train))
        data_saver = DataSaver()
        try:
            doc_id = data_saver.save(settings.MONGO_DNN_COLLECTION, doc=doc)
            logging.info("Successfully created the new document %s in the DB", doc_id)
        except Exception as e:
            logging.error("Failed to create a new document in the DB: %s", e)
        return self
