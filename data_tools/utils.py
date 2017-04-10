import json

import numpy as np


def one_hot_encoder(arr):
    levels = np.unique(arr)
    mat = np.eye(len(levels))
    d = {v: mat[:, j] for j, v in enumerate(levels)}
    return np.array([d[v] for v in arr])


class OneHotEncoder(object):
    def __init__(self):
        self._d = dict()
        self._levels = np.array([])

    @property
    def levels(self):
        return self._levels

    @property
    def n_levels(self):
        return len(self._levels)

    def fit(self, x):
        self._levels = np.unique(x)
        mat = np.eye(len(self._levels))
        self._d = {v: mat[:, j] for j, v in enumerate(self._levels)}
        return self

    def transform(self, x):
        return self._d[x]

    def to_json(self):
        return json.dumps({'_d': {k: list(v) for k, v in self._d.iteritems()}, '_levels': list(self._levels)})

    def from_json(self, json_obj):
        d = json.loads(json_obj)
        self._d = {k: np.array(v) for k, v in d.get('_d', {}).iteritems()}
        self._levels = np.array(d.get('_levels', []))
        return self
