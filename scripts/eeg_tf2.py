import argparse
import os

import settings
from classify.base_dnn import BaseDNN
from etc.dataset import build_data_sets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", choices=settings.SUBJECTS)
    args = parser.parse_args()

    file_name = os.path.join(settings.PATH_TO_MAT_FILES, args.subject.upper() + ".mat")
    ds = build_data_sets(file_name, avg_group_size=5, derivation='electric_field', random_state=42, test_proportion=0.2)

    model = BaseDNN(ds.n_channels, ds.trial_size, ds.n_classes)

    model.fit(ds)

    print "Complete."
