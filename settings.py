import logging
import os

from brainpy.eeg import EEG

PATH_TO_MAT_FILES = "/home/claudio/Projects/brain_data/vision/mat"
SUBJECTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
MAT_FILES = {s: os.path.join(PATH_TO_MAT_FILES, s.upper()) + ".mat" for s in SUBJECTS}
FILE_DESCRIPTION = [
    dict(subject='s1', n_trials=5188, n_samples=166016, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S1.mat'),
    dict(subject='s2', n_trials=5185, n_samples=165920, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S2.mat'),
    dict(subject='s3', n_trials=5186, n_samples=165952, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S3.mat'),
    dict(subject='s4', n_trials=5186, n_samples=165952, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S4.mat'),
    dict(subject='s5', n_trials=5185, n_samples=165920, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S5.mat'),
    dict(subject='s6', n_trials=5186, n_samples=165952, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S6.mat'),
    dict(subject='s7', n_trials=5188, n_samples=166016, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S7.mat'),
    dict(subject='s8', n_trials=5184, n_samples=165888, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S8.mat'),
    dict(subject='s9', n_trials=5185, n_samples=165920, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S9.mat'),
    dict(subject='s10', n_trials=5184, n_samples=165888, n_channels=124,
         filename='/home/claudio/Projects/brain_data/vision/mat/S10.mat')]

MONGO_DB = 'brain'
MONGO_PORT = 27017
MONGO_CHUNK_SIZE = 100000

MONGO_TEST_COLLECTION = 'coll_test'
MONGO_EEG_COLLECTION = 'coll_eeg'
MONGO_CLF_COLLECTION = 'coll_clf'
MONGO_ACC_COLLECTION = 'coll_acc'
MONGO_DNN_COLLECTION = 'coll_dnn'
MONGO_BATCH_COLLECTION = 'coll_batch'

MONGO_DEFAULT = dict(host='localhost', db=MONGO_DB, collection=MONGO_TEST_COLLECTION, port=MONGO_PORT,
                     chunk_size=MONGO_CHUNK_SIZE, drop_collections_on_load=True, transactions_collection='transactions',
                     transactions_source_csv_gz='transactions.csv.gz')

LOGGING_FILENAME = '/home/claudio/Projects/eeg_vision/logs/eeg_vision.log'
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'
LOGGING_BASIC_CONFIG = dict(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=LOGGING_FILENAME, filemode='a')

CHANNELS = range(124)
DERIVATIONS = EEG().DERIVATIONS
