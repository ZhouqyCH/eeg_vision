import numpy as np

from brainpy.eeg import EEG
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import settings
from etc.data_reader import data_reader

if __name__ == '__main__':
    filename = settings.MAT_FILES['s1']

    clf = LinearDiscriminantAnalysis()
    eeg = EEG(data_reader=data_reader).read(filename)
    eeg.average_trials(5, inplace=True)
    for ch in range(124):
        x_train, x_test, y_train, y_test = train_test_split(eeg.data[ch, :, :].squeeze().reshape((-1, eeg.trial_size)),
                                                            eeg.trial_labels.astype(np.int32), test_size=0.2,
                                                            random_state=42)
        pred = clf.fit(x_train, y_train).predict(x_test)
        print "channel %s: %.2f" % (ch, accuracy_score(y_test, pred))

    print "Complete"
