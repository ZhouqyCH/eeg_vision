class BaseTransform(object):
    def __init__(self):
        self.trained = False

    @property
    def params(self):
        p = {'trained': self.trained}
        p.update(self.params_internal)
        return p

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def params_internal(self):
        return {}

    def fit(self, x, y=None, **kwargs):
        self.fit_internal(x, y, **kwargs)
        self.trained = True
        return self

    def fit_internal(self, x, y, **kwargs):
        raise NotImplementedError()

    def transform(self, x):
        if not self.trained:
            raise RuntimeError("%s must be trained before calling transform" % self.name)
        return self.transform_internal(x)

    def transform_internal(self, x):
        raise NotImplementedError

    def get_params(self, deep=True):
        return self.params
