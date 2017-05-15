import argparse

import settings
from data_tools.bootstrap_batch import BatchCreator
from utils.logging_utils import logging_reconfig

logging_reconfig()
DEFAULT_WORK_DIR = "/home/claudio/Projects/brain_data/vision/batches"
DEFAULT_DERIVATION = "electric_field"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter_max", type=int, default=20000, help="maximum number of iterations")
    parser.add_argument("--avg_group_size", type=int, default=0, help="group size (# trials) to average the EEG data")
    parser.add_argument("-b", "--batch_size", type=int, help="the size of each batch", default=50)
    parser.add_argument("--subject", choices=settings.SUBJECTS, help="the subject to be used in this test")
    parser.add_argument("--workdir", type=str, default=DEFAULT_WORK_DIR, help="default working directory")
    parser.add_argument("--test_proportion", type=float, default=.15, help="the proportion of samples in the test set")
    parser.add_argument("-d", "--derivation", type=str, choices=settings.DERIVATIONS, default=DEFAULT_DERIVATION,
                        help="EEG derivation to be used")
    parser.add_argument("--seed", type=int, default=42, help="seed to set the random generator's state")
    args = parser.parse_args()
    work_dir = "/home/claudio/Projects/brain_data/vision/batches"
    bc = BatchCreator(args.subject, args.batch_size, args.avg_group_size, args.derivation, args.test_proportion,
                      args.workdir, seed=args.seed)
    bc.create(args.iter_max)
