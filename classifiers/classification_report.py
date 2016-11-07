from sklearn.metrics import accuracy_score, confusion_matrix


class ClassificationReport(object):
    def __init__(self, clf, dataset):
        self.clf = clf
        self.dataset = dataset

    def get_report(self):
        y_pred = self.clf.predict(self.dataset.train)
        train_accuracy = accuracy_score(self.dataset.train_labels, y_pred)
        train_confusion_matrix = confusion_matrix(self.dataset.train_labels, y_pred)
        y_pred = self.clf.predict(self.dataset.train)
        test_accuracy = accuracy_score(self.dataset.test_labels, y_pred)
        test_confusion_matrix = confusion_matrix(self.dataset.test_labels, y_pred)
        report = {
            'test_accuracy': test_accuracy,
            'train_accuracy': train_accuracy,
            'test_sample_size': len(self.dataset.test),
            'train_sample_size': len(self.dataset.train),
            'train_confusion_matrix': train_confusion_matrix,
            'test_confusion_matrix': test_confusion_matrix,
        }
        return report
