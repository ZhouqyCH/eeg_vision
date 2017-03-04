import argparse
import os

import settings
from dnn.dnn_models import DNN1
from etc.dataset import build_data_sets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", choices=settings.SUBJECTS)
    parser.add_argument("-g", "--avg_group_size", default=5, type=int)
    parser.add_argument("-d", "--derivation", choices=['potential', 'laplacian', 'electric_field'],
                        default='electric_field')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_proportion", type=float, default=0.2)
    parser.add_argument("--max_iter", type=int, default=500000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--display_step", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.75)
    args = parser.parse_args()

    design = dict(output=map(int, args.design_output.split('-')),
                  wc1=map(int, args.design_wc1.split('-')),
                  wc2=map(int, args.design_wc2.split('-')),
                  wd1=map(int, args.design_wd1.split('*')))

    file_name = os.path.join(settings.PATH_TO_MAT_FILES, args.subject.upper() + ".mat")
    ds = build_data_sets(file_name, avg_group_size=args.avg_group_size, derivation=args.derivation,
                         random_state=args.seed, test_proportion=args.test_proportion)

    model = DNN1(ds.n_channels, ds.trial_size, ds.n_classes, max_iter=args.max_iter, learning_rate=args.learning_rate,
                 batch_size=args.batch_size, display_step=args.display_step, dropout=args.dropout)
    model.fit(ds)
    print "Complete."
