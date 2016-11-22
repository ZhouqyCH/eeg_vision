import hashlib
import json

from brainpy.utils import json_default
from funcy import merge
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from base.base_data import BaseData


class BaseClassifier(BaseData):
    def __init__(self, random_state=42):
        super(BaseClassifier, self).__init__()
        self._clf = None
        self.random_state = random_state
        self._pipeline = None

    @property
    def pipeline(self):
        return Pipeline([('classifier', self.sklearn_classifier)])

    @property
    def pipeline_steps(self):
        return ','.join(map(lambda x: x[0], self.pipeline.steps))

    @property
    def sklearn_classifier(self):
        raise NotImplementedError

    @property
    def n_iter_search(self):
        return None

    @property
    def param_dist(self):
        return {}

    @property
    def doc(self):
        # TODO: replace with pipeline.get_params()
        clf_par = self.sklearn_classifier.get_params()
        clf_par.pop('estimator', None)
        return merge(dict(pipeline_steps=self.pipeline_steps, clf=self.name), clf_par)

    @property
    def identifier(self):
        word = json.dumps(self.doc, default=json_default)
        return hashlib.md5(word).hexdigest()

    def fit(self, x, y):
        self._pipeline = self.pipeline.fit(x, y)
        return self

    def predict(self, x):
        return self._pipeline.predict(x)

    def score(self, x, y):
        y_pred = self.predict(x)
        return dict(accuracy=accuracy_score(y, y_pred), confusion_matrix=confusion_matrix(y, y_pred).tolist(),
                    sample_size=len(y))
