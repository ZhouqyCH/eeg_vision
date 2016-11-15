import argparse

from brainpy.eeg import EEG

import settings
from base.classification_manager import ClassificationManager
from classifiers.anova_lda_classifier import AnovaLDAClassifier
from classifiers.anova_svm_classifier import AnovaSVMClassifier
from classifiers.lda_classifier import LDAClassifier
from classifiers.svm_classifier import SVMClassifier
from etc.data_reader import data_reader


# TODO: WORK ON TENSOR FLOW
from etc.eeg_to_db import eeg_to_db

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", choices=settings.SUBJECTS, default=settings.SUBJECTS[0])
    parser.add_argument("-g", "--group_size", type=int, default=1)
    parser.add_argument("-t", "--test_proportion", type=float, default=0.2)
    parser.add_argument("-r", "--random_seed", type=int, default=42)
    parser.add_argument("-c", "--channel_scheme", choices=['single', 'multi'], default='single')
    args = parser.parse_args()

    for derivation in ['potential', 'laplacian', 'electric field']:
        for subject, filename in settings.MAT_FILES.iteritems():
            eeg = EEG(data_reader=data_reader).read(filename)
            eeg.average_trials(args.group_size, inplace=True)
            if derivation == 'laplacian':
                eeg.get_laplacian(inplace=True)
            elif derivation == 'electric field':
                eeg.get_electric_field(inplace=True)
            eeg_to_db(eeg)
            clf = ClassificationManager([LDAClassifier(), SVMClassifier(), AnovaLDAClassifier(), AnovaSVMClassifier()],
                                        test_proportion=args.test_proportion,
                                        random_seed=args.random_seed, channel_scheme=args.channel_scheme)
            clf.eval(eeg, save_to_db=True)

    print "Complete."
