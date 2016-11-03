from classifiers.base_classifier import BaseClassifier
from sklearn import svm


class SVMClassifier(BaseClassifier):
    @property
    def clf(self):
        return svm.SVC()

    def get_params_json(self):
        d = super(SVMClassifier, self).get_params_json()
        d.update({
            'clf': 'SVC',
            'clf_C': self.clf.C,
            'clf_cache_size': self.clf.cache_size,
            'clf_class_weight': self.clf.class_weight,
            'clf_coef0': self.clf.coef0,
            'clf_degree': self.clf.degree,
            'clf_gamma': self.clf.gamma,
            'clf_kernel': self.clf.kernel,
            'clf_max_iter': self.clf.max_iter,
            'clf_random_state': self.random_state,
            'clf_shrinking': self.clf.shrinking,
            'clf_tol': self.clf.tol})
        return d
