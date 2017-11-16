import numpy as np
import pandas as pd
import xgboost as xgb

import dask.array as da
from dask.array.utils import assert_eq
import dask.dataframe as dd
from distributed import Client
from distributed.utils_test import gen_cluster, loop, cluster  # noqa

import dask_xgboost as dxgb

df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   'y': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]})
labels = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 1, 1])

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}

X = df.values
y = labels.values


def test_classifier(loop):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            a = dxgb.XGBClassifier()
            X2 = da.from_array(X, 5)
            y2 = da.from_array(y, 5)
            a.fit(X2, y2)
            p1 = a.predict(X2)

    b = xgb.XGBClassifier()
    b.fit(X, y)
    np.testing.assert_array_almost_equal(a.feature_importances_,
                                         b.feature_importances_)
    assert_eq(p1, b.predict(X))


def test_regressor(loop):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            a = dxgb.XGBRegressor()
            X2 = da.from_array(X, 5)
            y2 = da.from_array(y, 5)
            a.fit(X2, y2)
            p1 = a.predict(X2)

    b = xgb.XGBRegressor()
    b.fit(X, y)
    assert_eq(p1, b.predict(X))


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
def test_dmatrix_kwargs(c, s, a, b):
    xgb.rabit.init()  # workaround for "Doing rabit call after Finalize"
    dX = da.from_array(X, chunks=(2, 2))
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb._train(c, param, dX, dy, {"missing": 0.0})

    # Distributed model matches local model with dmatrix kwargs
    dtrain = xgb.DMatrix(X, label=y, missing=0.0)
    bst = xgb.train(param, dtrain)
    result = bst.predict(dtrain)
    dresult = dbst.predict(dtrain)
    assert np.abs(result - dresult).sum() < 0.02

    # Distributed model gives bad predictions without dmatrix kwargs
    dtrain_incompat = xgb.DMatrix(X, label=y)
    dresult_incompat = dbst.predict(dtrain_incompat)
    assert np.abs(result - dresult_incompat).sum() > 0.02


@gen_cluster(client=True, timeout=None)
def test_numpy(c, s, a, b):
    xgb.rabit.init()  # workaround for "Doing rabit call after Finalize"
    dX = da.from_array(X, chunks=(2, 2))
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb._train(c, param, dX, dy)
    dbst = yield dxgb._train(c, param, dX, dy)  # we can do this twice

    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(param, dtrain)

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


def test_synchronous_api(loop):  # noqa
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
