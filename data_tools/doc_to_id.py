import hashlib
import json

from data_tools.json_default import json_default


def doc_to_id(doc):
    word = json.dumps(doc, default=json_default)
    return hashlib.md5(word).hexdigest()
