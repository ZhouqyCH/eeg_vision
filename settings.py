import logging
import os

PATH_TO_MAT_FILES = "/home/claudio/Projects/brain_data/vision/mat"
SUBJECTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
MAT_FILES = {s: os.path.join(PATH_TO_MAT_FILES, s.upper()) + ".mat" for s in SUBJECTS}

MONGO = dict(host='localhost', db='brain', collection='eeg', port=27017, chunk_size=100000,
             drop_collections_on_load=True, transactions_collection='transactions',
             transactions_source_csv_gz='transactions.csv.gz')

LOGGING_FILENAME = '/home/claudio/Projects/eeg_vision/logs/eeg_vision.log'
LOGGING_LEVEL = logging.INFO
LOGGING_FORMAT = '%(asctime)s %(levelname)s %(message)s'
LOGGING_BASIC_CONFIG = dict(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                            filename=LOGGING_FILENAME, filemode='a')
