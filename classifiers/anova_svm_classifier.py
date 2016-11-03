from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from classifiers.base_classifier import BaseClassifier


class AnovaSVMClassifier(BaseClassifier):
    @property
    def pipeline(self):
        anova_filter = SelectKBest(f_regression, k=5)
        return Pipeline([('anova', anova_filter), ('svc', SVC())])
