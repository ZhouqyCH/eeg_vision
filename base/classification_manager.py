import logging

import settings
from base.base_data import BaseData
from base.train_test_splitter import TrainTestSplitter

logging.basicConfig(**settings.LOGGING_BASIC_CONFIG)


class ClassificationManager(BaseData):
    def __init__(self, classifiers, **kwargs):
        super(ClassificationManager, self).__init__()
        self.splitter = TrainTestSplitter(**kwargs)
        self.classifiers = classifiers
        self._params = self.splitter.params
        self._report = []

    def get_params_json(self):
        return self._params

    def eval(self, eeg):
        self.splitter.eval(eeg)
        self._report = []
        for clf in self.classifiers:
            for ds in self.splitter.datasets:
                report = clf.eval(ds)
                self._report.append(report)
                logging.info("%s - %s: test accuracy: %s" % (clf.name, ds.name, report["test_accuracy"]))
        return self
