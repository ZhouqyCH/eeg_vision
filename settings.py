import os

PATH = "/home/claudio/Projects/brain_data/vision/mat"
SUBJECTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
FILES = {s: os.path.join(PATH, s.upper()) + ".mat" for s in SUBJECTS}

MONGO = {
        'host': 'localhost',
        'db': 'brain',
        'port': 27017,
        'chunk_size': 100000,
        'drop_collections_on_load': True,
        'transactions_collection': 'transactions',
        'transactions_source_csv_gz': 'transactions.csv.gz'
    }