import json
from sklearn.metrics import accuracy_score, confusion_matrix

from brainpy.utils import json_default

from base.base_data import BaseData


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
    def pipeline_steps(self):
        return ','.join(map(lambda x: x[0], self.pipeline.steps))

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

    def fit(self, ds):
        self._pipeline = self.pipeline.fit(ds.train, ds.train_labels)
        return self

    def predict(self, x):
        return self._pipeline.predict(x)

    def score(self, ds):
        y_pred_train = self.predict(ds.train)
        y_pred_test = self.predict(ds.test)
        result = {'test_accuracy': accuracy_score(ds.test_labels, y_pred_test),
                  'train_accuracy': accuracy_score(ds.train_labels, y_pred_train),
                  'test_sample_size': len(ds.test),
                  'train_sample_size': len(ds.train),
                  'train_confusion_matrix': confusion_matrix(ds.train_labels, y_pred_train).tolist(),
                  'test_confusion_matrix': confusion_matrix(ds.test_labels, y_pred_test).tolist(),
                  'classifier': self.name,
                  'dataset': ds.name,
                  'pipeline_steps': self.pipeline_steps,
                  'clf_params': self.classifier_params}
        return result
