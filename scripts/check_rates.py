import settings
from base.mongo_io import MongoIO

import pandas as pd


if __name__ == '__main__':

    pd.set_option("display.width", 10000)

    data = MongoIO(collection=settings.MONGO_ACC_COLLECTION).load(return_cursor=False)
    clf = MongoIO(collection=settings.MONGO_CLF_COLLECTION).load(return_cursor=False)
    clf_id2name = dict(pd.DataFrame(clf)[['_id', 'classifier']].values.tolist())
    data = pd.DataFrame(data)
    data['classifier'] = data['clf_id'].apply(lambda x: clf_id2name[x])
    data['method'] = data.apply(lambda x: '%s_%s' % (x['derivation'], x['classifier'][:-10]), axis=1)
    max_rates = data.pivot_table(index='subject', columns='method', values='accuracy', aggfunc='max')
    max_rates['max_rate'] = max_rates.apply(lambda x: x.max(), axis=1)
    max_rates['best_method'] = max_rates.drop('max_rate', axis=1).apply(lambda x: x.argmax(), axis=1)
    max_rates['worst_method'] = max_rates.drop(['max_rate', 'best_method'], axis=1).apply(lambda x: x.argmin(), axis=1)
    print max_rates
