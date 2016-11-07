from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline

from classifiers.base_classifier import BaseClassifier
from classifiers.merge_components import MergeComponents


class AnovaLDAClassifier(BaseClassifier):
    @property
    def pipeline(self):
        anova_filter = SelectKBest(f_regression, k=5)
        return Pipeline([('merge_comps', MergeComponents()), ('anova', anova_filter),
                         ('classifier', LinearDiscriminantAnalysis())])
