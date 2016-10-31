from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from classifiers.base_classifier import BaseClassifier


class LDAClassifier(BaseClassifier):
    @property
    def clf(self):
        return LinearDiscriminantAnalysis()
