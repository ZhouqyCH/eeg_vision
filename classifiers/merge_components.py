from classifiers.base_transform import BaseTransform


class MergeComponents(BaseTransform):
    def __init__(self):
        super(MergeComponents, self).__init__()
        self.has_comps = False

    def fit_internal(self, x, y, **kwargs):
        self.has_comps = x.squeeze().ndim > 2
        return self

    def transform_internal(self, x):
        if self.has_comps:
            return x.squeeze().reshape((x.shape[0], -1))
        return x.squeeze()

    @property
    def params_internal(self):
        return {'has_comps': self.has_comps}
