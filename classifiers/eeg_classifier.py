import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from classifiers.base_classifier import BaseClassifier


class EEGClassifier(BaseClassifier):
    def __init__(self, eeg, **kwargs):
        super(EEGClassifier, self).__init__(**kwargs)
        self.eeg = eeg

    # TODO: INCLUDE PIPELINES
    def classify(self, verbose=True):
        for ch in range(self.eeg.n_channels):
            x_train, x_test, y_train, y_test = train_test_split(self.eeg.data[ch, :].reshape((-1, self.eeg.trial_size)),
                                                                self.eeg.trial_labels.astype(np.int32),
                                                                test_size=self.test_proportion,
                                                                random_state=self.random_state)
            self.clf.fit(x_train, y_train)
            prediction = self.clf.predict(x_test)
            self.accuracy.loc[ch, "score"] = accuracy_score(y_test, prediction)
            r = pd.DataFrame({"actual": y_test.ravel(), "prediction": prediction})
            r["channel"] = ch
            self.result = pd.concat((self.result, r), ignore_index=True)
            if verbose:
                print "Channel %s: %.0f%%" % (ch, 100 * self.accuracy[ch])
        return self

    def get_params_internal(self):
        d = self.clf.get_params_for_json()
        d.update(self.eeg.get_params_for_json())
        return d
