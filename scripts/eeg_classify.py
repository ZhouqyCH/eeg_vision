import argparse

from brainpy.eeg import EEG

import settings
from base.train_test_splitter import TrainTestSplitter
from classifiers.anova_svm_classifier import AnovaSVMClassifier
from etc.data_reader import data_reader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", choices=settings.SUBJECTS, default=settings.SUBJECTS[0])
    parser.add_argument("-g", "--group_size", type=int, default=1)
    parser.add_argument("-t", "--test_proportion", type=float, default=0.2)
    parser.add_argument("-r", "--random_seed", type=int, default=42)
    parser.add_argument("-c", "--channel_scheme", choices=['single', 'multi'], default='single')
    args = parser.parse_args()

    file_name = settings.FILES[args.subject]
    eeg = EEG(data_reader=data_reader).read(file_name)
    eeg.average_trials(args.group_size, inplace=True)
    eeg.get_electric_field(inplace=True)
    eeg = TrainTestSplitter(test_proportion=args.test_proportion,
                            random_seed=args.random_seed,
                            channel_scheme='single').eval(eeg)
    clf = AnovaSVMClassifier().fit(eeg)
    clf.test(eeg)
    # eeg.save()
    print clf.test_accuracy

    print "Complete."
