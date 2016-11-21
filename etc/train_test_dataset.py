import numpy as np
from sklearn.cross_validation import train_test_split


def train_test_dataset(data, labels, test_proportion, dataset_name="no_name", random_seed=42):
    x_train, x_test, y_train, y_test = train_test_split(data.squeeze(), np.asarray(labels, dtype=np.int32),
                                                        test_size=test_proportion, random_state=random_seed)
    return type("Dataset", (), dict(name=dataset_name, train=x_train, test=x_test, train_labels=y_train,
                                    test_labels=y_test))
