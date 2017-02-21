import numpy as np
import pandas as pd
from tornado import gen
import xgboost as xgb

import dask.array as da
import dask.dataframe as dd
from distributed import Client, Nanny
from distributed.utils_test import gen_cluster, loop, cluster

import dask_xgboost as dxgb

df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'y': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})
labels = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 1, 1])

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

X = df.values
y = labels.values


@gen_cluster(client=True, timeout=None)
def test_basic(c, s, a, b):
    dtrain = xgb.DMatrix(df, label=labels)
    bst = xgb.train(param, dtrain)

    ddf = dd.from_pandas(df, npartitions=4)
    dlabels = dd.from_pandas(labels, npartitions=4)
    dbst = yield dxgb._train(c, param, ddf, dlabels)
    dbst = yield dxgb._train(c, param, ddf, dlabels)  # we can do this twice

    result = bst.predict(dtrain)
    dresult = dbst.predict(dtrain)

    correct = (result > 0.5) == labels
    dcorrect = (dresult > 0.5) == labels
    assert dcorrect.sum() >= correct.sum()

    predictions = dxgb.predict(c, dbst, ddf)
    assert isinstance(predictions, dd.Series)
    predictions = yield c.compute(predictions)._result()
    assert isinstance(predictions, pd.Series)

    assert ((predictions > 0.5) != labels).sum() < 2


@gen_cluster(client=True, timeout=None)
def test_numpy(c, s, a, b):
    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(param, dtrain)

    dX = da.from_array(X, chunks=(2, 2))
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb._train(c, param, dX, dy)
    dbst = yield dxgb._train(c, param, dX, dy)  # we can do this twice

    result = bst.predict(dtrain)
    dresult = dbst.predict(dtrain)

    correct = (result > 0.5) == y
    dcorrect = (dresult > 0.5) == y
    assert dcorrect.sum() >= correct.sum()

    predictions = dxgb.predict(c, dbst, dX)
    assert isinstance(predictions, da.Array)
    predictions = yield c.compute(predictions)._result()
    assert isinstance(predictions, np.ndarray)

    assert ((predictions > 0.5) != labels).sum() < 2


def test_synchronous_api(loop):
    dtrain = xgb.DMatrix(df, label=labels)
    bst = xgb.train(param, dtrain)

    ddf = dd.from_pandas(df, npartitions=4)
    dlabels = dd.from_pandas(labels, npartitions=4)

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:

            dbst = dxgb.train(c, param, ddf, dlabels)

            result = bst.predict(dtrain)
            dresult = dbst.predict(dtrain)

            correct = (result > 0.5) == labels
            dcorrect = (dresult > 0.5) == labels
            assert dcorrect.sum() >= correct.sum()
