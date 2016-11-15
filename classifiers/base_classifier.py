import json

from base.base_data import BaseData
from utils.json_default import json_default


class BaseClassifier(BaseData):
    def __init__(self, random_state=42):
        super(BaseClassifier, self).__init__()
        self._clf = None
        self.random_state = random_state
        self._pipeline = None

    @property
    def pipeline(self):
        raise NotImplementedError

    @property
    def pipeline_str(self):
        return ', '.join(self.pipeline.steps)

    @property
    def sklearn_classifier(self):
        raise NotImplementedError

    @property
    def classifier_params(self):
        return self.sklearn_classifier.get_params()

    def get_params_json(self):
        return {
            'pipeline': json.dumps(self.pipeline.get_params(deep=True), default=json_default),
            'random_state': self.random_state,
            'classifier': self.name}

    def fit(self, dataset):
        self._pipeline = self.pipeline.fit(dataset.train, dataset.train_labels)
        return self

    def predict(self, x):
        return self._pipeline.predict(x)

    def score(self, x, y):
        return self._pipeline.score(self, x, y)
