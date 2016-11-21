from funcy import merge
from sklearn.metrics import accuracy_score, confusion_matrix

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
    def doc(self):
        return merge(dict(pipeline_steps=self.pipeline_steps, clf=self.name), self.sklearn_classifier.get_params())

    def fit(self, x, y):
        self._pipeline = self.pipeline.fit(x, y)
        return self

    def predict(self, x):
        return self._pipeline.predict(x)

    def score(self, x, y):
        y_pred = self.predict(x)
        return dict(accuracy=accuracy_score(y, y_pred), confusion_matrix=confusion_matrix(y, y_pred).tolist(),
                    sample_size=len(y))
