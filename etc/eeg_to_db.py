from base.mongo_io import MongoIO


def eeg_to_db(eeg):
    """"Save the parameters of the eeg signal on the database"""
    data = eeg.get_params()
    data['_id'] = eeg.identifier
    mongo = MongoIO(collection='eeg_data')
    mongo.save(data)
