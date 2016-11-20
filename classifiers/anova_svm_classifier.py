from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from classifiers.base_classifier import BaseClassifier


class AnovaSVMClassifier(BaseClassifier):
    @property
    def sklearn_classifier(self):
        return SVC()

    @property
    def pipeline(self):
        anova_filter = SelectKBest(f_regression, k=20)
        return Pipeline([('anova', anova_filter), ('classifier', self.sklearn_classifier)])
