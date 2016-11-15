from sklearn.metrics import accuracy_score, confusion_matrix


class ClassificationResult(object):
    def __init__(self, clf, dataset):
        self.clf = clf
        self.dataset = dataset

    def get_result(self):
        y_pred_train = self.clf.predict(self.dataset.train)
        y_pred_test = self.clf.predict(self.dataset.test)
        return dict(dict(classifier=self.clf.name,
                    test_accuracy=accuracy_score(self.dataset.test_labels, y_pred_test),
                    train_accuracy=accuracy_score(self.dataset.train_labels, y_pred_train),
                    test_sample_size=len(self.dataset.test), train_sample_size=len(self.dataset.train),
                    train_confusion_matrix=confusion_matrix(self.dataset.train_labels, y_pred_train),
                    test_confusion_matrix=confusion_matrix(self.dataset.test_labels, y_pred_test)),
                    **self.dataset.get_params())
