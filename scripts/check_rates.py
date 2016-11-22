import settings
from base.mongo_io import MongoIO

if __name__ == '__main__':
    db = MongoIO(collection=settings.MONGO_CLF_COLLECTION)
    db.load(return_cursor=False, criteria={'subject': 's1'})
