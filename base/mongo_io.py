from funcy import merge
from pymongo import MongoClient

import settings


class MongoIO(object):
    def __init__(self, **kwargs):
        options = merge(settings.MONGO_DEFAULT, kwargs)
        self.client = MongoClient(host=options['host'], port=options['port'], connect=False)
        self.db = self.client[options['db']]
        self.collection = self.db[options['collection']]

    def save(self, doc):
        return self.collection.insert_one(doc).inserted_id

    def remove(self, id):
        return self.collection.remove(id)

    def load(self, return_cursor=False, criteria=None, projection=None):
        if criteria is None:
            criteria = {}
        if projection is None:
            cursor = self.collection.find(criteria)
        else:
            cursor = self.collection.find(criteria, projection)

        # Return a cursor for large amounts of data
        if return_cursor:
            return cursor
        else:
            return [item for item in cursor]
