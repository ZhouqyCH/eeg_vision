import logging
import os
import random
import sys
from collections import defaultdict

import deepdish as dd
import numpy as np
import pandas as pd
from brainpy.eeg import EEG
from sklearn.cross_validation import train_test_split

import settings
from data_saver import DataSaver
from matlab_data_reader import matlab_data_reader, get_matlab_labels
from .utils import OneHotEncoder


class BatchIterator(object):
    def __init__(self, batch_size, obj_list):
        self._obj_list = list(obj_list)
        self._batch_size = batch_size
        assert batch_size < len(self._obj_list)
        self._index_in_cycle = 0
        self._index_max = len(self._obj_list)

    def next_batch(self):
        if self._index_in_cycle + self._batch_size > self._index_max:
            random.shuffle(self._obj_list)
            self._index_in_cycle = 0
        start = self._index_in_cycle
        end = start + self._batch_size
        self._index_in_cycle = end
        return self._obj_list[start:end]


class BatchCreator(object):
    def __init__(self, batch_size, outdir, avg_group_size=1, eeg_derivation='electric_field', test_proportion=0.15,
                 seed=42, subject="s1"):
        assert eeg_derivation in ['potential', 'electric_field', 'laplacian'], \
            "Derivation '%s' is not supported" % eeg_derivation
        random.seed(seed)
        eeg_info = filter(lambda x: x['subject'] == subject, settings.FILE_LIST)[0]
        labels = get_matlab_labels(eeg_info['filename'])
        classes = list(set(labels))
        self._info = {'batch_size': batch_size, 'test_proportion': test_proportion, 'outdir': outdir, 'seed': seed,
                      'avg_group_size': avg_group_size, 'eeg_derivation': eeg_derivation.lower(), 'subject': subject,
                      'eeg': eeg_info, 'labels': labels, 'classes': classes, 'n_class': len(classes)}
        self.label_encoder = OneHotEncoder().fit(labels)

    def _get_batches(self, max_iter, values_list, file_prefix, file_extension):
        random.seed(self._info['seed'])
        batch_iter = BatchIterator(self._info['batch_size'], values_list)
        batches = defaultdict(list)
        for n, _ in enumerate(range(0, max_iter, self._info['batch_size'])):
            batch_file = os.path.join(self._info['outdir'], "%s_%s%s" % (file_prefix, n, file_extension))
            curr_batch = batch_iter.next_batch()
            for trial_index in curr_batch:
                batches[batch_file].append(trial_index)
        return batches

    def create(self, max_iter):
        logging.info("Processing the EEG file %s", self._info['eeg']['filename'])
        eeg = EEG(data_reader=matlab_data_reader).read(self._info['eeg']['filename'])
        if self._info['avg_group_size'] > 1:
            logging.info("Averaging trials")
            eeg.average_trials(self._info['avg_group_size'], inplace=True)
        if self._info['eeg_derivation'] != 'potential':
            logging.info("Building the %s derivation", self._info['eeg_derivation'])
            if self._info['eeg_derivation'] == "electric_field":
                eeg.get_electric_field(inplace=True)
            elif self._info['eeg_derivation'] == 'laplacian':
                eeg.get_laplacian(inplace=True)
            else:
                raise KeyError("Derivation '%s' is not supported", self._info['eeg_derivation'])
        if eeg.data.ndim == 3:
            eeg.data = eeg.data[:, :, :, np.newaxis]
        labels = eeg.trial_labels
        eeg = eeg.data.reshape(self._info['eeg']['n_channels'], self._info['eeg']['trial_size'], -1, 3).transpose((2, 0, 1, 3))
        idx_train, idx_test = train_test_split(range(len(labels)),
                                               test_size=self._info['test_proportion'],
                                               random_state=self._info['seed'])
        self._info['train_size'] = len(idx_train)
        self._info['test_size'] = len(idx_test)
        batches_train = self._get_batches(max_iter, idx_train, "%s_train" % self._info['subject'], '.hd5')
        self._info['n_train_batches'] = len(batches_train)
        self._info['files_train'] = batches_train.keys()
        created_files = defaultdict(dict)
        logging.info("Creating the batch files")
        for batch_file, trial_indices in batches_train.iteritems():
            encoded_labels = map(lambda x: self.label_encoder.transform(labels[x]), trial_indices)
            if created_files['train'].get(batch_file):
                rec = dd.io.load(batch_file)
                samples = np.r_[rec['samples'], eeg[trial_indices, :, :, :]]
                labs = rec['labels'].append(encoded_labels)
            else:
                samples = eeg[trial_indices, :, :, :]
                labs = encoded_labels
            try:
                dd.io.save(batch_file, {'samples': samples, 'labels': labs})
                if not created_files['train'].get(batch_file):
                    logging.info("Successfully created the training file %s", batch_file)
                created_files['train'][batch_file] = True
            except Exception as e:
                logging.error("Failed to create the training file %s: %s", batch_file, e)
                sys.exit(1)
        test_file = os.path.join(self._info['outdir'], "%s_test.hd5" % self._info['subject'])
        self._info['n_test_batches'] = 1
        self._info['files_test'] = [test_file]
        created_files['test'][test_file] = True
        try:
            dd.io.save(test_file, {'samples': eeg[idx_test, :, :, :],
                                   'labels': [list(self.label_encoder.transform(x)) for x in labels[idx_test]]})
            logging.error("Successfully created the test file %s", test_file)
        except Exception as e:
            logging.error("Failed to create the test file %s: %s", test_file, e)
            sys.exit(1)
        logging.info("Finished to create batch files")
        doc = pd.Series(self._info).to_dict()
        data_saver = DataSaver()
        try:
            doc_id = data_saver.save(settings.MONGO_DNN_COLLECTION, doc=doc)
            logging.info("Successfully created the new document %s in the DB", doc_id)
        except Exception as e:
            logging.error("Failed to create a new document in the DB: %s", e)
        return self
