import gc
import logging
import os
import sys
import traceback

import deepdish as dd
from funcy import merge
from pymongo import MongoClient

import settings
from data_tools.bootstrap_batch import BootstrapBatch
from dnn.convnet import ConvNet
from utils.logging_utils import logging_reconfig


if __name__ == '__main__':
    import argparse

    logging_reconfig()

    parser = argparse.ArgumentParser()
    parser.add_argument("subject", choices=settings.SUBJECTS)
    parser.add_argument("model_output", type=str)
    parser.add_argument("-d", "--derivation", default='electric_field')
    parser.add_argument("--group_size_max", type=int, default=10)
    parser.add_argument("--batch_count", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.info("Starting to train the CNN model for Brainwave Classification")
    client = MongoClient('localhost', 27017)
    db = client.brain

    train_info = db.train_info.find_one({'subject': args.subject})
    if not train_info:
        logging.info("Failed to load the train info for this subject from the DB")
        sys.exit(0)
    logging.info("Successfully loaded the train info from the DB")

    ann_config = db.ann_config.find_one({'name': 'ann_simple', 'n_conv_layers': 2, 'n_fc_layers': 2, 'n_classes': 6})
    if not ann_config:
        logging.info("Failed to load the ANN configuration for this subject from the DB")
        sys.exit(0)
    logging.info("Successfully loaded the ANN configuration from the DB")

    try:
        data = dd.io.load(train_info['path'])
        logging.info("Successfully loaded the data: %s samples", data['n_samples'])
    except Exception as e:
        logging.info("Failed to load the data: %s", e)
        sys.exit(0)

    logging.info("Creating temporary batch files")
    try:
        work_dir = os.path.join(os.path.dirname(train_info['path']), "batches")
        batcher = BootstrapBatch(data['samples'], data['labels'], args.group_size_max, args.batch_size, seed=args.seed,
                                 auto_remove_files=True)\
            .create(args.batch_count, work_dir, prefix='%s_train_' % train_info['subject'])
        logging.info("Successfully created the batch files")
    except Exception as e:
        logging.info("Failed to create the batch files: %s", e)
        sys.exit(0)

    del data
    gc.collect()

    logging.info("Starting to train the neural network")
    conv_net = ConvNet(ann_config['W_conv1'], ann_config['bias_conv1'], ann_config['W_conv2'], ann_config['bias_conv2'],
                       ann_config['W_fc1'], ann_config['bias_fc1'], ann_config['W_fc2'], ann_config['bias_fc2'],
                       train_info['trial_size'], train_info['n_comps'], train_info['n_channels'],
                       train_info['n_classes'], learning_rate=ann_config['learning_rate'], )
    conv_net.train(batcher, output_filename=args.model_output)

    try:
        ann_config.pop("_id")
        doc = merge(ann_config, {"train_accuracy": conv_net.train_accuracy, "path": args.model_output,
                                 "batch_size": args.batch_size, "group_size_max": args.group_size_max,
                                 "seed": args.seed, "batch_count": args.batch_count})
        doc = db.trained_models.insert_one(doc)
        logging.info("Successfully created the database entry %s for the result", doc['_id'])
    except Exception, e:
        logging.info("Failed to create a database entry for the result:\n%s\n%s", e, traceback.format_exc())
    finally:
        batcher.remove_batch_files()
        logging.info("Successfully deleted the temporary batch files from the disc")
    logging.info("Finished to train the model")
