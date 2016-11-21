from brainpy.utils import json_default
from funcy import merge


def save_to_db(db, doc, identifier=None):
    """"Save the parameters of the eeg signal to the database"""
    doc = doc.copy()
    for key in doc.keys():
        if not isinstance(doc[key], dict):
            doc[key] = json_default(doc[key])
        else:
            doc[key] = {k: json_default(doc[key][k])for k in doc[key].keys()}
    if identifier:
        doc = merge(doc, dict(_id=identifier))
        db.remove(identifier)
    db.save(doc)
