import logging

SUBJECTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
DEFAULT_WORK_DIR = "/home/claudio/Projects/brain_data/vision/"


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
