from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from classification.base_classifier import BaseClassifier


class LDAClassifier(BaseClassifier):
    @property
    def classifier(self):
        return LinearDiscriminantAnalysis()


class LRClassifier(BaseClassifier):
    @property
    def param_dist(self):
        return dict(C=[0.5, 1.0, 5.], penalty=['l1', 'l2'])

    @property
    def classifier(self):
        return LogisticRegression(random_state=self.random_state)


class SVMClassifier(BaseClassifier):
    @property
    def param_dist(self):
        return dict(C=[0.5, 1.0, 5.], kernel=['linear', 'rbf'], shrinking=[True, False], probability=[True, False])

    @property
    def classifier(self):
        return SVC(random_state=self.random_state)


class RFClassifier(BaseClassifier):
    @property
    def param_dist(self):
        return dict(max_depth=[3, 4, 5, None], bootstrap=[True, False], criterion=["gini", "entropy"],
                    n_estimators=range(20, 120, 10))

    @property
    def classifier(self):
        return RandomForestClassifier(random_state=self.random_state)
