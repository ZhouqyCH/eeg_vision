from datetime import datetime

from base.mongo_io import MongoIO


class BaseData(MongoIO):
    @property
    def name(self):
        return self.__class__.__name__

    def _get_params(self):
        d = {'class_name': self.name,
             'update_time': datetime.now()}
        d.update(self.get_params_json())
        return d

    def get_params_json(self):
        return {}

    def to_mongo(self):
        dta = self._get_params()
        self.save(dta)
        return self
