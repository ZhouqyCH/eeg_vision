from scipy.stats import randint as sp_randint

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.grid_search import RandomizedSearchCV
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


class RFClassifier(BaseClassifier):
    @property
    def param_dist(self):
        return dict(max_depth=[3, None], max_features=sp_randint(1, 11), min_samples_split=sp_randint(1, 11),
                    min_samples_leaf=sp_randint(1, 11), bootstrap=[True, False], criterion=["gini", "entropy"],
                    n_estimators=range(20, 120, 10))

    @property
    def n_iter_search(self):
        return 20

    @property
    def sklearn_classifier(self):
        return RandomizedSearchCV(RandomForestClassifier(), param_distributions=self.param_dist,
                                  n_iter=self.n_iter_search)
