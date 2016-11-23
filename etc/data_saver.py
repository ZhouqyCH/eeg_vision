from base.mongo_io import MongoIO
from etc.doc_to_id import doc_to_id
from etc.json_default import json_default


class DataSaver(object):
    def __init__(self):
        self.db = dict()

    def save(self, collection, doc=None, identifier=None, modify_object=False, replace_existing=True):
        if doc is None:
            return
        if not isinstance(doc, dict):
            raise TypeError("%s: doc should be a dictionary got type %s" % (self.__class__.__name__, type(doc)))
        if collection not in self.db.keys():
            self.db[collection] = MongoIO(collection=collection)
        coll = self.db[collection]
        if not modify_object:
            doc = doc.copy()
        if "_id" not in doc:
            if identifier:
                doc["_id"] = identifier
            else:
                doc["_id"] = doc_to_id(doc)
        if replace_existing:
            coll.remove(doc["_id"])
        elif coll.findOne({"_id": doc["_id"]}):
            return doc["_id"]
        for key in doc.keys():
            if isinstance(doc[key], dict):
                doc[key] = {k: json_default(doc[key][k]) for k in doc[key].keys()}
            elif key != "_id":
                doc[key] = json_default(doc[key])
        _id = coll.save(doc)
        return _id
