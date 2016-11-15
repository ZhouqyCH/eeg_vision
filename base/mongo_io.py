from pymongo import MongoClient
import settings


class MongoIO(object):
    def __init__(self, collection=None):
        self.conn = {'host': settings.MONGO['host'], 'port': settings.MONGO['port']}
        self.client = MongoClient(**self.conn)
        self.db = self.client[settings.MONGO['db']]
        if not collection:
            collection = settings.MONGO['collection']
        self.collection = self.db[collection]

    def save(self, data):
        return self.collection.insert(data)

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
