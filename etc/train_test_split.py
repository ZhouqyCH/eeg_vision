import gc
import numpy as np
from etc.dataset import Dataset


def train_test_split(data, labels, test_proportion, random_seed, suffix):
    np.random.rand(random_seed)
    n_labels = len(labels)
    index = np.random.rand(n_labels).argsort()
    test_size = int(test_proportion * n_labels)
    train_labels = labels[test_size:]
    test_labels = labels[:test_size]
    datasets = []
    if data.ndim == 3:
        for n in range(data.shape[2]):
            gc.collect()
            ds = Dataset("%s_%s" % (suffix, n), data[index[test_size:], :, n].squeeze(), train_labels,
                         data[index[:test_size], :, n].squeeze(), test_labels)
            datasets.append(ds)
    elif data.ndim == 2:
        ds = Dataset(suffix, data[index[test_size:], :], train_labels, data[index[:test_size], :], test_labels)
        datasets = [ds]
    else:
        raise RuntimeError("train_test_split: number of dimensions must be 2 or 3")
    return datasets
