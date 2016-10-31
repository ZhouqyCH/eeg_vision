import pandas as pd
from sklearn.metrics import accuracy_score

from base.base_data import BaseData


class BaseClassifier(BaseData):
    def __init__(self, db='eeg', mango_collection='class_coll', random_state=42, test_proportion=0.2):
        super(BaseClassifier, self).__init__(db, mango_collection)
        self._clf = None
        self.random_state = random_state
        self.test_proportion = test_proportion
        self.result = pd.DataFrame([])
        self.accuracy = pd.DataFrame([])

    @property
    def clf(self):
        return None

    @property
    def name(self):
        return self.clf.__class__.__name__

    def get_params(self):
        return {
            'classifier': self.name,
            'random_state': self.random_state,
            'test_proportion': self.test_proportion,
            "result": self.result.to_json(),
            "accuracy": self.accuracy.to_json()}

    def fit(self, x, y):
        self._clf = self.clf
        self._clf.fit(x, y)
        return self

    def predict(self, y):
        return self._clf.predict(y)

    def accuracy_score(self, y_true, y_pred, normalize=True, sample_weight=None):
        self.accuracy = accuracy_score(y_true, y_pred, normalize=normalize, sample_weight=sample_weight)
        return self
