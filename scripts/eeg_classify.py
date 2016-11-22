import argparse
import logging

from brainpy.eeg import EEG
from funcy import merge

import settings
from classify.classifiers import LDAClassifier, SVMClassifier, LRClassifier
from etc.data_reader import data_reader
from etc.data_saver import DataSaver
from etc.train_test_dataset import train_test_dataset
from utils.logging_utils import logging_reconfig

logging_reconfig()

# TODO: WORK ON TENSOR FLOW
# TODO: SAVE CLASSIFIERS ON THE DATABASE
CLASSIFIERS = \
    {
        "lda": LDAClassifier(),
        "svm": SVMClassifier(),
        "logreg": LRClassifier(),
        # "rf": RFClassifier()
    }


def valid_proportion(p):
    if not isinstance(p, float) or p <= 0 or p >= 1:
        raise argparse.ArgumentTypeError("Proportion must be a float number greater than 0 and less than 1")
    else:
        return p


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", nargs="*", choices=settings.SUBJECTS + ['all'], default=['all'])
    parser.add_argument("--channels", nargs="*", choices=map(str, settings.CHANNELS) + ['all'], default=['all'])
    parser.add_argument("-d", "--derivation", nargs="*", choices=settings.DERIVATIONS + ['all'], default=['all'])
    parser.add_argument("--classifier", nargs="*", choices=CLASSIFIERS.keys() + ["all"], default=['all'])
    parser.add_argument("--group_size", type=int, default=0)
    parser.add_argument("--single_channels", type=bool, default=True)
    parser.add_argument("--test_proportion", type=valid_proportion, default=0.2)
    parser.add_argument("-r", "--random_seed", type=int, default=42)
    parser.add_argument("--save", type=bool, default=True)
    args = parser.parse_args()

    if 'all' in args.subject:
        sub2file = settings.MAT_FILES.copy()
    else:
        sub2file = {s: settings.MAT_FILES[s] for s in args.subject}

    if 'all' in args.derivation:
        derivations = list(settings.DERIVATIONS)
    else:
        derivations = args.derivation

    if 'all' in args.channels:
        channels = settings.CHANNELS
    else:
        channels = map(int, args.channels)

    if 'all' in args.classifier:
        classifiers = CLASSIFIERS.values()
    else:
        classifiers = [CLASSIFIERS[c] for c in args.classifier]

    data_saver = DataSaver()

    for subject, filename in sub2file.iteritems():
        for derivation in derivations:

            eeg = EEG(data_reader=data_reader).read(filename)
            if args.group_size:
                eeg.average_trials(args.group_size, inplace=True)

            if derivation == 'laplacian':
                eeg.get_laplacian(inplace=True)
            elif derivation == 'electric_field':
                eeg.get_electric_field(inplace=True)

            if args.save:
                data_saver.save(settings.MONGO_EEG_COLLECTION, eeg.doc, identifier=eeg.identifier)
                logging.info("Successfully saved info on the DB: %s %s _id=%s" % (subject, derivation, eeg.identifier))

            logging.info("Generating datasets for classification")
            if args.single_channels:
                datasets = []
                for ch in channels:
                    ds = train_test_dataset(eeg.to_clf_format(ch), eeg.trial_labels, args.test_proportion,
                                            random_seed=args.random_seed, dataset_name="channel_%s" % ch)
                    datasets.append(ds)
            else:
                datasets = [train_test_dataset(eeg.to_clf_format(channels), eeg.trial_labels, args.test_proportion,
                                               random_seed=args.random_seed,
                                               dataset_name="channel_%s" % '_'.join(args.channels))]

            for ds in datasets:
                for clf in classifiers:
                    data_saver.save(settings.MONGO_CLF_COLLECTION, clf.doc, identifier=clf.identifier)
                    score_doc = clf.fit(ds.train, ds.train_labels).score(ds.test, ds.test_labels)
                    logging.info(
                        "%s %s %s %s acc: %.2f" % (subject, derivation, ds.name, clf.name, score_doc['accuracy']))
                    if args.save:
                        doc = merge({'subject': subject, 'dataset': ds.name, 'group_size': args.group_size,
                                     'derivation': derivation, 'eeg_id': eeg.identifier, 'clf_id': clf.identifier},
                                    score_doc)
                        data_saver.save(settings.MONGO_ACC_COLLECTION, doc)

    logging.info("Complete")
