import argparse

from data_tools.batch_creator import BatchCreator
from utils.logging_utils import logging_reconfig

logging_reconfig()
DEFAULT_WORK_DIR = "/home/claudio/Projects/brain_data/vision/batches"
DEFAULT_DERIVATION = "electric_field"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("iter_max", type=int, help="maximum number of iterations")
    parser.add_argument("batch_size", type=int, help="the size of each batch")
    parser.add_argument("--workdir", type=str, default=DEFAULT_WORK_DIR, help="default working directory")
    parser.add_argument("-d", "--derivation", type=str, default=DEFAULT_DERIVATION, help="EEG derivation to be used")
    args = parser.parse_args()
    work_dir = "/home/claudio/Projects/brain_data/vision/batches"
    bc = BatchCreator(args.batch_size, args.workdir, eeg_derivation=args.derivation)
    bc.create(args.iter_max)
