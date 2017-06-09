import datetime
from pymongo import MongoClient


if __name__ == '__main__':
    print "Save ANN config in the DB"
    client = MongoClient('localhost', 27017)
    db = client.brain

    # First convolutional layer:
    # Computes 32 features for each 5x5 patch. Its weight tensor will have a shape of
    # [5, 5, 3, 32]. The first two dimensions are the patch size, the next is the number of electric field components,
    # and the last is the number of output components

    # Second Convolutional Layer.
    # 64 features for each 5x5 patch

    # Densely Connected Layer.
    # A fully-connected layer with 1024 neurons. The tensor from the pooling layer is reshaped into a batch of
    # vectors, multiplied by a weight matrix, added to a bias, and applied to a ReLU

    doc = {'W_conv1': [5, 5, 3, 32], 'bias_conv1': [32], 'W_conv2': [5, 5, 32, 64], 'bias_conv2': [64],
           'W_fc1': [31 * 8 * 64, 1024], 'bias_fc1': [1024], 'W_fc2': [1024, 6], 'bias_fc2': [6], 'learning_rate': 1e-3,
           'updated_time': datetime.datetime.utcnow(), 'n_conv_layers': 2, 'n_fc_layers': 2, 'n_classes': 6,
           'name': 'ann_simple'}
    ret = db.ann_config.insert_one(doc)
    print "Successfully created the DB entry %s" % ret.inserted_id
