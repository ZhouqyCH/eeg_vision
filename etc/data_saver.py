from brainpy.utils import json_default
from funcy import merge

from base.mongo_io import MongoIO


class DataSaver(object):
    def __init__(self):
        self.db = dict()

    def save(self, collection, doc, identifier=None, modify_object=False):
        if collection not in self.db.keys():
            self.db[collection] = MongoIO(collection=collection)
        if not modify_object:
            doc = doc.copy()
        for collection in doc.keys():
            if not isinstance(doc[collection], dict):
                doc[collection] = json_default(doc[collection])
            else:
                doc[collection] = {k: json_default(doc[collection][k]) for k in doc[collection].keys()}
        if identifier:
            doc = merge(doc, dict(_id=identifier))
            self.db[collection].remove(identifier)
        self.db[collection].save(doc)
