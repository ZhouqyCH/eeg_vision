from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from classifiers.base_classifier import BaseClassifier


class SVMClassifier(BaseClassifier):
    @property
    def sklearn_classifier(self):
        return SVC()

    @property
    def pipeline(self):
        return Pipeline([('classifier', self.sklearn_classifier)])
