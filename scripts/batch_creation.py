from data_tools.batch_creator import BatchCreator
from utils.logging_utils import logging_reconfig

if __name__ == '__main__':
    logging_reconfig()
    b_size = 50
    n_iter = 100000
    work_dir = "/home/claudio/Projects/brain_data/vision/batches"
    bc = BatchCreator(b_size, work_dir)
    bc.create(n_iter)
