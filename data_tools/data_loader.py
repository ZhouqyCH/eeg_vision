from base.mongo_io import MongoIO


class DataLoader(object):
    @staticmethod
    def load(collection, _id=None):
        coll = MongoIO(collection=collection)
        if _id:
            return coll.load(criteria={'_id': _id})
        return coll.load()[-1]
