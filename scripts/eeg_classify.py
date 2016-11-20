import argparse
import logging

from brainpy.eeg import EEG
from funcy import merge

import settings
from base.mongo_io import MongoIO
from base.save_to_db import save_to_db
from classifiers.anova_lda_classifier import AnovaLDAClassifier
from classifiers.anova_svm_classifier import AnovaSVMClassifier
from classifiers.lda_classifier import LDAClassifier
from classifiers.svm_classifier import SVMClassifier
from etc.data_reader import data_reader
from etc.eeg_reshape import eeg_reshape
from etc.train_test_split import train_test_split
from utils.logging_utils import logging_reconfig

# TODO: WORK ON TENSOR FLOW

CLASSIFIERS = {
    "anova_lda": AnovaLDAClassifier(),
    "anova_svm": AnovaSVMClassifier(),
    "lda": LDAClassifier(),
    "svm": SVMClassifier()}


def valid_proportion(p):
    if not isinstance(p, float) or p <= 0 or p >= 1:
        raise argparse.ArgumentTypeError("Proportion must be a float number greater than 0 and less than 1")
    else:
        return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", nargs="*", choices=settings.SUBJECTS + ['all'], default=['all'])
    parser.add_argument("-g", "--group_size", type=int, default=0)
    parser.add_argument("-c", "--channel_scheme", nargs="*", choices=settings.CHANNEL_SCHEMES + ['all'],
                        default=['all'])
    parser.add_argument("-d", "--derivation", nargs="*", choices=settings.DERIVATIONS + ['all'], default=['all'])
    parser.add_argument("--classifier", nargs="*", choices=CLASSIFIERS.keys() + ["all"], default=['all'])
    parser.add_argument("-p", "--test_proportion", type=valid_proportion, default=0.2)
    parser.add_argument("-r", "--random_seed", type=int, default=42)
    args = parser.parse_args()

    if 'all' in args.subject:
        sub2file = settings.MAT_FILES.copy()
    else:
        sub2file = {s: settings.MAT_FILES[s] for s in args.subject}

    if 'all' in args.derivation:
        derivations = list(settings.DERIVATIONS)
    else:
        derivations = args.derivation

    if 'all' in args.channel_scheme:
        channel_scheme = list(settings.CHANNEL_SCHEMES)
    else:
        channel_scheme = args.channel_scheme

    if 'all' in args.classifier:
        classifiers = CLASSIFIERS.values()
    else:
        classifiers = [CLASSIFIERS[c] for c in args.classifier]

    db = MongoIO(collection=settings.MONGO_EEG_CLF_COLLECTION)
    vars_args = vars(args)
    logging_reconfig()

    for subject, filename in sub2file.iteritems():
        for derivation in derivations:

            eeg = EEG(data_reader=data_reader).read(filename)
            if args.group_size:
                eeg.average_trials(args.group_size, inplace=True)

            if derivation == 'laplacian':
                eeg.get_laplacian(inplace=True)
            elif derivation == 'electric_field':
                eeg.get_electric_field(inplace=True)

            save_to_db(eeg.doc, collection=settings.MONGO_EEG_DATA_COLLECTION, identifier=eeg.identifier)
            logging.info("Successfully stored data from subject %s - %s - on the DB" % (subject, derivation))

            for cs in channel_scheme:
                datasets = train_test_split(eeg_reshape(eeg.data, eeg.trial_size, cs == 'single'), eeg.trial_labels,
                                            args.test_proportion, args.random_seed, cs)
                for ds in datasets:
                    for clf in classifiers:
                        score = clf.fit(ds).score(ds)
                        logging.info("%s, %s, %s, %s, acc: %s" % (subject, derivation, ds.name, clf.name,
                                                                  score['test_accuracy']))
                        db.save(merge(vars_args, {'eeg_id': eeg.identifier, 'derivation': derivation}, score))
    logging.info("Complete")
