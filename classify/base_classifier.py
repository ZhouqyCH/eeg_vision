import hashlib
import json

from brainpy.utils import json_default
from funcy import merge
from sklearn.grid_search import RandomizedSearchCV
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
    def transformers(self):
        return []

    @property
    def pipeline(self):
        if self.param_dist:
            classifier = RandomizedSearchCV(self.classifier, param_distributions=self.param_dist,
                                            n_iter=self.n_iter_search, random_state=self.random_state)
        else:
            classifier = self.classifier
        return Pipeline(self.transformers + [('classifier', classifier)])

    @property
    def pipeline_steps(self):
        return ','.join(map(lambda x: x[0], self.pipeline.steps))

    @property
    def n_iter_search(self):
        return 20

    @property
    def classifier(self):
        raise NotImplementedError

    @property
    def param_dist(self):
        return {}

    @property
    def doc(self):
        clf_par = self.pipeline.get_params()
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
