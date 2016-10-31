import json

import io
import os


class JsonIO(object):
    def __init__(self, path, filename, extension='json'):
        self.path = path
        self.filename = filename
        self.extension = extension
        # self.file_io = os.path.join(dir_name, '.'.join((base_ filename, filename_suffix)))

    def save(self, data):
        if os.path.isfile(self.full_name):
            mode = 'a' # Append existing file
        else:
            mode = 'w' # Create a new file
        with io.open(self.full_name, mode, encoding='utf-8') as f:
            f.write(unicode(json.dumps(data, ensure_ascii=False)))

    def load(self):
        with io.open(self.full_name, encoding='utf-8') as f:
            return f.read()

    @property
    def full_name(self):
        return '{0}/{1}.{2}'.format(self.path, self.filename, self.extension)
