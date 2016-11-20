from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from classifiers.base_classifier import BaseClassifier


class LDAClassifier(BaseClassifier):
    @property
    def sklearn_classifier(self):
        return LinearDiscriminantAnalysis()

    @property
    def pipeline(self):
        return Pipeline([('classifier', self.sklearn_classifier)])
