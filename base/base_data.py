from datetime import datetime

from base.mongo_io import MongoIO


class BaseData(MongoIO):
    def __init__(self):
        super(BaseData, self).__init__(self.mango_collection)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def mango_collection(self):
        return 'default'

    def _get_params(self):
        d = {'class_name': self.name,
             'update_time': datetime.now()}
        d.update(self.get_params())
        return d

    # this method should be overwritten by subclasses
    def get_params(self):
        return {}

    def to_mongo(self):
        dta = self._get_params()
        self.save(dta)
        return self
