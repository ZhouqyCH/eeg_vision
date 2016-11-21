from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from classify.base_classifier import BaseClassifier


class LDAClassifier(BaseClassifier):
    @property
    def sklearn_classifier(self):
        return LinearDiscriminantAnalysis()

    @property
    def pipeline(self):
        return Pipeline([('classifier', self.sklearn_classifier)])


class AnovaLDAClassifier(BaseClassifier):
    @property
    def sklearn_classifier(self):
        return LinearDiscriminantAnalysis()

    @property
    def pipeline(self):
        anova_filter = SelectKBest(f_regression, k=20)
        return Pipeline([('anova', anova_filter), ('classifier', self.sklearn_classifier)])


class SVMClassifier(BaseClassifier):
    @property
    def sklearn_classifier(self):
        return SVC()

    @property
    def pipeline(self):
        return Pipeline([('classifier', self.sklearn_classifier)])


class AnovaSVMClassifier(BaseClassifier):
    @property
    def sklearn_classifier(self):
        return SVC()

    @property
    def pipeline(self):
        anova_filter = SelectKBest(f_regression, k=20)
        return Pipeline([('anova', anova_filter), ('classifier', self.sklearn_classifier)])
