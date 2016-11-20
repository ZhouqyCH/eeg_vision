from brainpy.utils import json_default

import settings
from base.mongo_io import MongoIO


def save_to_db(doc, collection=None, identifier=None):
    """"Save the parameters of the eeg signal to the database"""
    doc = doc.copy()
    if collection is None:
        collection = settings.MONGO_EEG_DATA_COLLECTION
    for key in doc.keys():
        if not isinstance(doc[key], dict):
            doc[key] = json_default(doc[key])
        else:
            doc[key] = {k: json_default(doc[key][k])for k in doc[key].keys()}
    db = MongoIO(collection=collection)
    if identifier:
        doc['_id'] = identifier
        db.remove(identifier)
    db.save(doc)
    return db.db.name
