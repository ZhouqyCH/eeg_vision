import datetime
from funcy import merge
from pymongo import MongoClient


if __name__ == '__main__':
    print "Saving file info in the DB"
    client = MongoClient('localhost', 27017)
    db = client.brain
    dt = datetime.datetime.utcnow()
    file_info = [
        dict(subject='s1', n_trials=5188, n_samples=166016, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S1.mat'),
        dict(subject='s2', n_trials=5185, n_samples=165920, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S2.mat'),
        dict(subject='s3', n_trials=5186, n_samples=165952, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S3.mat'),
        dict(subject='s4', n_trials=5186, n_samples=165952, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S4.mat'),
        dict(subject='s5', n_trials=5185, n_samples=165920, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S5.mat'),
        dict(subject='s6', n_trials=5186, n_samples=165952, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S6.mat'),
        dict(subject='s7', n_trials=5188, n_samples=166016, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S7.mat'),
        dict(subject='s8', n_trials=5184, n_samples=165888, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S8.mat'),
        dict(subject='s9', n_trials=5185, n_samples=165920, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S9.mat'),
        dict(subject='s10', n_trials=5184, n_samples=165888, n_channels=124, trial_size=32,
             path='/home/claudio/Projects/brain_data/vision/mat/S10.mat')]
    for doc in file_info:
        ret = db.mat_info.insert_one(merge(doc, {'updated_time': dt}))
        print "Successfully saved file info for subject '%s' in the DB: uid %s" % (doc['subject'], ret.inserted_id)
    print "Finished saving data info in the DB"
