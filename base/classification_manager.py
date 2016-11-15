import logging

from funcy import merge

import settings
from base.base_data import BaseData
from base.train_test_splitter import TrainTestSplitter
from classifiers.classification_result import ClassificationResult

logging.basicConfig(**settings.LOGGING_BASIC_CONFIG)
logging.getLogger().addHandler(logging.StreamHandler())


class ClassificationManager(BaseData):
    def __init__(self, classifiers, **kwargs):
        super(ClassificationManager, self).__init__()
        self.splitter = TrainTestSplitter(**kwargs)
        self.classifiers = classifiers
        self._params = {}
        self._result = []

    def eval(self, eeg, save_to_db=False):
        self.splitter.eval(eeg)
        splitter_params = self.splitter.params
        for clf in self.classifiers:
            for ds in self.splitter.datasets:
                clf.fit(ds)
                r = ClassificationResult(clf, ds).get_result()
                r = merge(r, splitter_params, {'classifier': clf.name, 'dataset': ds.name, 'eeg_id': eeg.identifier,
                                               'derivation': eeg.derivation, 'pipeline_steps': clf.pipeline_steps,
                                               'clf_random_state': clf.random_state,
                                               'classifier_params': clf.classifier_params})
                logging.info("%s: EEG %s - %s %s: %s" % (self.name, eeg.derivation, clf.name, ds.name, r['test_accuracy']))
                if save_to_db:
                    self.save(r)
                    logging.info("%s: Successfully stored the result on the DB" % self.name)
        return self
