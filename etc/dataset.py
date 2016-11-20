class Dataset(object):
    def __init__(self, name, train, train_labels, test, test_labels):
        self.name = name
        self.train = train
        self.train_labels = train_labels
        self.test = test
        self.test_labels = test_labels
