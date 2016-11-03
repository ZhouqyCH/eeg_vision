import json

import pandas as pd
from sklearn.metrics import accuracy_score

from base.base_data import BaseData
from utils.json_default import json_default


class BaseClassifier(BaseData):
    def __init__(self, random_state=42, test_proportion=0.2):
        super(BaseClassifier, self).__init__()
        self._clf = None
        self.random_state = random_state
        self.test_proportion = test_proportion
        self.test_predictions = pd.DataFrame([])
        self.test_accuracy = None
        self._pipeline = None
        self.predictions = []

    @property
    def mango_collection(self):
        return 'class_coll'

    @property
    def pipeline(self):
        raise NotImplementedError

    def get_params_json(self):
        return {
            'pipeline': json.dumps(self.pipeline.get_params(deep=True), default=json_default),
            'random_state': self.random_state,
            'test_proportion': self.test_proportion,
            "test_predictions": self.test_predictions,
            "test_accuracy": self.test_accuracy}

    def fit(self, data_obj):
        self._pipeline = self.pipeline.fit(data_obj.train, data_obj.train_labels)
        return self

    def test(self, data_obj):
        self.test_predictions = self._pipeline.predict(data_obj.test)
        self.test_accuracy = accuracy_score(data_obj.test_labels, self.test_predictions)
        return self
