import numpy as np
import pandas as pd
import xgboost as xgb
import sparse
import scipy.sparse

import pytest

import dask
import dask.array as da
from dask.array.utils import assert_eq
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.datasets import load_iris
from distributed.utils_test import gen_cluster, loop, cluster  # noqa

import dask_xgboost as dxgb

# Workaround for conflict with distributed 1.23.0
# https://github.com/dask/dask-xgboost/pull/27#issuecomment-417474734
from concurrent.futures import ThreadPoolExecutor
import distributed.comm.utils

distributed.comm.utils._offload_executor = ThreadPoolExecutor(max_workers=2)


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


def test_multiclass_classifier(loop):  # noqa
    # data
    iris = load_iris()
    X, y = iris.data, iris.target
    dX = da.from_array(X, 5)
    dy = da.from_array(y, 5)
    df = pd.DataFrame(X, columns=iris.feature_names)
    labels = pd.Series(y, name='target')

    ddf = dd.from_pandas(df, 2)
    dlabels = dd.from_pandas(labels, 2)
    # model
    a = xgb.XGBClassifier()  # array
    b = dxgb.XGBClassifier()
    c = xgb.XGBClassifier()  # frame
    d = dxgb.XGBClassifier()

    with cluster() as (s, [_, _]):
        with Client(s['address'], loop=loop):
            # fit
            a.fit(X, y)  # array
            b.fit(dX, dy, classes=[0, 1, 2])
            c.fit(df, labels)  # frame
            d.fit(ddf, dlabels, classes=[0, 1, 2])

            # check
            da.utils.assert_eq(a.predict(X), b.predict(dX))
            da.utils.assert_eq(a.predict_proba(X), b.predict_proba(dX))
            da.utils.assert_eq(c.predict(df), d.predict(ddf))
            da.utils.assert_eq(c.predict_proba(df), d.predict_proba(ddf))


@pytest.mark.parametrize("kind", ['array', 'dataframe'])  # noqa
def test_classifier_multi(kind, loop):

    if kind == 'array':
        X2 = da.from_array(X, 5)
        y2 = da.from_array(
            np.array([0, 1, 2, 0, 1, 2, 0, 0, 0, 1]),
            chunks=5,
        )
    else:
        X2 = dd.from_pandas(df, npartitions=2)
        y2 = dd.from_pandas(labels, npartitions=2)

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            a = dxgb.XGBClassifier(num_class=3, n_estimators=10,
                                   objective="multi:softprob")
            a.fit(X2, y2)
            p1 = a.predict(X2)

            assert dask.is_dask_collection(p1)

            if kind == 'array':
                assert p1.shape == (10,)

            result = p1.compute()
            assert result.shape == (10,)

            # proba
            p2 = a.predict_proba(X2)
            assert dask.is_dask_collection(p2)

            if kind == 'array':
                assert p2.shape == (10, 3)
            assert p2.compute().shape == (10, 3)


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


@gen_cluster(client=True, timeout=None, check_new_threads=False)
def test_basic(c, s, a, b):
    dtrain = xgb.DMatrix(df, label=labels)
    bst = xgb.train(param, dtrain)

    ddf = dd.from_pandas(df, npartitions=4)
    dlabels = dd.from_pandas(labels, npartitions=4)
    dbst = yield dxgb.train(c, param, ddf, dlabels)
    dbst = yield dxgb.train(c, param, ddf, dlabels)  # we can do this twice

    result = bst.predict(dtrain)
    dresult = dbst.predict(dtrain)

    correct = (result > 0.5) == labels
    dcorrect = (dresult > 0.5) == labels
    assert dcorrect.sum() >= correct.sum()

    predictions = dxgb.predict(c, dbst, ddf)
    assert isinstance(predictions, da.Array)
    predictions = yield c.compute(predictions)._result()
    assert isinstance(predictions, np.ndarray)

    assert ((predictions > 0.5) != labels).sum() < 2


@gen_cluster(client=True, timeout=None, check_new_threads=False)
def test_dmatrix_kwargs(c, s, a, b):
    xgb.rabit.init()  # workaround for "Doing rabit call after Finalize"
    dX = da.from_array(X, chunks=(2, 2))
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb.train(c, param, dX, dy, {"missing": 0.0})

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


def _test_container(dbst, predictions, X_type):
    dtrain = xgb.DMatrix(X_type(X), label=y)
    bst = xgb.train(param, dtrain)

    result = bst.predict(dtrain)
    dresult = dbst.predict(dtrain)

    correct = (result > 0.5) == y
    dcorrect = (dresult > 0.5) == y

    assert dcorrect.sum() >= correct.sum()
    assert isinstance(predictions, np.ndarray)
    assert ((predictions > 0.5) != labels).sum() < 2


@gen_cluster(client=True, timeout=None, check_new_threads=False)
def test_numpy(c, s, a, b):
    xgb.rabit.init()  # workaround for "Doing rabit call after Finalize"
    dX = da.from_array(X, chunks=(2, 2))
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb.train(c, param, dX, dy)
    dbst = yield dxgb.train(c, param, dX, dy)  # we can do this twice

    predictions = dxgb.predict(c, dbst, dX)
    assert isinstance(predictions, da.Array)
    predictions = yield c.compute(predictions)
    _test_container(dbst, predictions, np.array)


@gen_cluster(client=True, timeout=None, check_new_threads=False)
def test_scipy_sparse(c, s, a, b):
    xgb.rabit.init()  # workaround for "Doing rabit call after Finalize"
    dX = da.from_array(X, chunks=(2, 2)).map_blocks(scipy.sparse.csr_matrix)
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb.train(c, param, dX, dy)
    dbst = yield dxgb.train(c, param, dX, dy)  # we can do this twice

    predictions = dxgb.predict(c, dbst, dX)
    assert isinstance(predictions, da.Array)

    predictions_result = yield c.compute(predictions)
    _test_container(dbst, predictions_result, scipy.sparse.csr_matrix)


@gen_cluster(client=True, timeout=None, check_new_threads=False)
def test_sparse(c, s, a, b):
    xgb.rabit.init()  # workaround for "Doing rabit call after Finalize"
    dX = da.from_array(X, chunks=(2, 2)).map_blocks(sparse.COO)
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb.train(c, param, dX, dy)
    dbst = yield dxgb.train(c, param, dX, dy)  # we can do this twice

    predictions = dxgb.predict(c, dbst, dX)
    assert isinstance(predictions, da.Array)

    predictions_result = yield c.compute(predictions)
    _test_container(dbst, predictions_result, sparse.COO)


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


@gen_cluster(client=True, timeout=None, check_new_threads=False)
def test_errors(c, s, a, b):
    def f(part):
        raise Exception('foo')

    df = dd.demo.make_timeseries()
    df = df.map_partitions(f, meta=df._meta)

    with pytest.raises(Exception) as info:
        yield dxgb.train(c, param, df, df.x)

    assert 'foo' in str(info.value)
