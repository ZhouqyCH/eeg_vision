import argparse
import logging
import os

import datetime
import deepdish as dd

import numpy as np

from brainpy.eeg import EEG
from funcy import merge
from pymongo import MongoClient
from sklearn.cross_validation import train_test_split

import settings
from data_tools.data_tools import matlab_data_reader
from utils.logging_utils import logging_reconfig

logging_reconfig()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs='*', choices=settings.SUBJECTS, default=settings.SUBJECTS,
                        help="a list of specific subjects")
    parser.add_argument("--workdir", type=str, default=settings.DEFAULT_WORK_DIR, help="working directory")
    parser.add_argument("--test_proportion", type=float, default=.15, help="the proportion of samples in the test set")
    parser.add_argument("-d", "--derivation", type=str, choices=['potential', 'laplacian', 'electric_field'],
                        default='electric_field', help="EEG derivation to be used")
    parser.add_argument("--seed", type=int, default=42, help="seed to set the random number generator")
    args = parser.parse_args()

    logging.info("Splitting EEG data into training and test sets")

    client = MongoClient('localhost', 27017)
    db = client.brain

    prefix = "TrainTestSplitter"
    updated_time = datetime.datetime.utcnow()
    for cnt, subject in enumerate(args.subjects):
        logging.info("%s: %s of %s - processing subject %s", prefix, cnt+1, len(args.subjects), subject)
        file_info = db.eeg.find_one({"type": "file_info", "subject": "s1"})
        logging.info("%s: reading EEG data", prefix)
        eeg = EEG(data_reader=matlab_data_reader).read(file_info['path'])
        labels = eeg.trial_labels
        n_classes = len(set(labels))
        logging.info("%s: %s labels and %s classes", prefix, len(labels), n_classes)
        if args.derivation == 'potential':
            eeg = eeg.data
            n_comps = 1
        elif args.derivation == 'laplacian':
            logging.info("%s: estimating the Laplacian derivation", prefix)
            eeg = eeg.get_laplacian(inplace=True).data
            n_comps = 1
        else:
            logging.info("%s: estimating the electric field derivation", prefix)
            eeg = eeg.get_electric_field(inplace=True).data[:, :, :, np.newaxis]
            n_comps = 3
        # TODO: only works for the electric field derivation
        logging.info("%s: reshaping the data", prefix)
        eeg = eeg.reshape(file_info['n_channels'], file_info['trial_size'], -1, 3).transpose((2, 0, 1, 3))

        logging.info("%s: splitting data into training and test sets", prefix)
        train_samples, test_samples, train_labels, test_labels = train_test_split(eeg,
                                                                                  labels,
                                                                                  test_size=args.test_proportion,
                                                                                  random_state=args.seed)
        base_info = {'source': file_info['path'], 'n_channels': file_info['n_channels'], 'subject': subject,
                     'trial_size': file_info['trial_size'], 'updated_time': updated_time, 'n_comps': n_comps,
                     'derivation': args.derivation, 'n_classes': n_classes, 'source_id': file_info['_id']}

        train_file = os.path.join(args.workdir, "%s_train.hd5" % subject)
        logging.info("%s: saving training data to %s", prefix, train_file)
        dd.io.save(train_file, merge(base_info, {'samples': train_samples, 'labels': train_labels,
                                                 'n_samples': len(train_labels)}))
        doc = merge(base_info, {"path": train_file, "n_samples": len(train_labels)})
        obj = db.train_info.insert_one(doc)
        logging.info("%s: successfully created a new DB entry: _id %s", prefix, obj.inserted_id)

        test_file = os.path.join(args.workdir, "%s_test.hd5" % subject)
        logging.info("%s: saving test data to %s", prefix, test_file)
        dd.io.save(test_file, merge(base_info, {'samples': test_samples, 'labels': test_labels,
                                                'n_samples': len(test_labels)}))
        doc = merge(base_info, {"path": train_file, "n_samples": len(test_labels)})
        obj = db.test_info.insert_one(doc)
        logging.info("%s: successfully created a new DB entry: _id %s", prefix, obj.inserted_id)

    logging.info("%s: complete.", prefix)
