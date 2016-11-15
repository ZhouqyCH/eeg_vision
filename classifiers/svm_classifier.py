from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from classifiers.base_classifier import BaseClassifier
from classifiers.merge_components import MergeComponents


class SVMClassifier(BaseClassifier):
    @property
    def sklearn_classifier(self):
        return SVC()

    @property
    def pipeline(self):
        return Pipeline(
            [('merge_comps', MergeComponents()), ('classifier', self.sklearn_classifier)])
