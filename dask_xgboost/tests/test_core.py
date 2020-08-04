# Workaround for conflict with distributed 1.23.0
# https://github.com/dask/dask-xgboost/pull/27#issuecomment-417474734
from concurrent.futures import ThreadPoolExecutor

import dask
import dask.array as da
import dask.dataframe as dd
import distributed.comm.utils
import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import xgboost as xgb
from dask.array.utils import assert_eq
from dask.distributed import Client
from distributed.utils_test import cluster, gen_cluster, loop  # noqa
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split

import dask_xgboost as dxgb
from dask_xgboost.core import _package_evals

distributed.comm.utils._offload_executor = ThreadPoolExecutor(max_workers=2)


df = pd.DataFrame(
    {"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "y": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],}
)
labels = pd.Series([1, 0, 1, 0, 1, 0, 1, 1, 1, 1])

param = {
    "max_depth": 2,
    "eta": 1,
    "silent": 1,
    "objective": "binary:logistic",
}

X = df.values
y = labels.values


def test_classifier(loop):  # noqa
    digits = load_digits(2)
    X = digits["data"]
    y = digits["target"]

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            a = dxgb.XGBClassifier()
            X2 = da.from_array(X)
            y2 = da.from_array(y)
            a.fit(X2, y2)
            p1 = a.predict(X2)

    b = xgb.XGBClassifier()
    b.fit(X, y)
    np.testing.assert_array_almost_equal(a.feature_importances_, b.feature_importances_)
    assert_eq(p1, b.predict(X))


def test_classifier_different_chunks(loop):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            a = dxgb.XGBClassifier()
            X2 = da.from_array(X, 5)
            y2 = da.from_array(y, 4)

            with pytest.raises(ValueError):
                a.fit(X2, y2)


def test_multiclass_classifier(loop):  # noqa
    # data
    iris = load_iris()
    X, y = iris.data, iris.target
    dX = da.from_array(X, 5)
    dy = da.from_array(y, 5)
    df = pd.DataFrame(X, columns=iris.feature_names)
    labels = pd.Series(y, name="target")

    ddf = dd.from_pandas(df, 2)
    dlabels = dd.from_pandas(labels, 2)
    # model
    a = xgb.XGBClassifier()  # array
    b = dxgb.XGBClassifier()
    c = xgb.XGBClassifier()  # frame
    d = dxgb.XGBClassifier()

    with cluster() as (s, [_, _]):
        with Client(s["address"], loop=loop):
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


def test_classifier_early_stopping(loop):  # noqa
    # data
    digits = load_digits(2)
    X = digits["data"]
    y = digits["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    dX_train = da.from_array(X_train)
    dy_train = da.from_array(y_train)

    clf1 = dxgb.XGBClassifier()
    clf2 = dxgb.XGBClassifier()
    clf3 = dxgb.XGBClassifier()
    with cluster() as (s, [_, _]):
        with Client(s["address"], loop=loop):
            clf1.fit(
                dX_train,
                dy_train,
                early_stopping_rounds=5,
                eval_metric="auc",
                eval_set=[(X_test, y_test)],
            )
            clf2.fit(
                dX_train,
                dy_train,
                early_stopping_rounds=4,
                eval_metric="auc",
                eval_set=[(X_test, y_test)],
            )

            # should be the same
            assert clf1.best_score == clf2.best_score
            assert clf1.best_score != 1

            # check overfit
            clf3.fit(
                dX_train,
                dy_train,
                early_stopping_rounds=10,
                eval_metric="auc",
                eval_set=[(X_test, y_test)],
            )
            assert clf3.best_score == 1


def test_package_evals():
    # data
    digits = load_digits(2)
    X = digits["data"]
    y = digits["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    evals = _package_evals([(X_test, y_test), (X, y_test)])

    assert len(evals) == 2

    evals = _package_evals(
        [(X_test, y_test), (X, y_test)], sample_weight_eval_set=[[1], [2]]
    )

    assert len(evals) == 2

    evals = _package_evals(
        [(X_test, y_test), (X, y_test)], sample_weight_eval_set=[[1]]
    )

    assert len(evals) == 1


def test_validation_weights_xgbclassifier(loop):  # noqa
    from sklearn.datasets import make_hastie_10_2

    # prepare training and test data
    X, y = make_hastie_10_2(n_samples=2000, random_state=42)
    labels, y = np.unique(y, return_inverse=True)

    param_dist = {
        "objective": "binary:logistic",
        "n_estimators": 2,
        "random_state": 123,
    }

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            X_train, X_test = X[:1600], X[1600:]
            y_train, y_test = y[:1600], y[1600:]

            dX_train = da.from_array(X_train)
            dy_train = da.from_array(y_train)

            # instantiate model
            clf = dxgb.XGBClassifier(**param_dist)

            # train it using instance weights only in the training set
            weights_train = np.random.choice([1, 2], len(X_train))
            weights_train = da.from_array(weights_train)
            clf.fit(
                dX_train,
                dy_train,
                sample_weight=weights_train,
                eval_set=[(X_test, y_test)],
                eval_metric="logloss",
            )

            # evaluate logloss metric on test set *without* using weights
            evals_result_without_weights = clf.evals_result()
            logloss_without_weights = evals_result_without_weights["validation_0"][
                "logloss"
            ]

            # now use weights for the test set
            np.random.seed(0)
            weights_test = np.random.choice([1, 2], len(X_test))
            clf.fit(
                dX_train,
                dy_train,
                sample_weight=weights_train,
                eval_set=[(X_test, y_test)],
                sample_weight_eval_set=[weights_test],
                eval_metric="logloss",
            )
            evals_result_with_weights = clf.evals_result()
            logloss_with_weights = evals_result_with_weights["validation_0"]["logloss"]

    # check that the logloss in the test set is actually different
    # when using weights than when not using them
    assert all((logloss_with_weights[i] != logloss_without_weights[i] for i in [0, 1]))


@pytest.mark.parametrize("kind", ["array", "dataframe"])
def test_classifier_multi(kind, loop):  # noqa: F811

    if kind == "array":
        X2 = da.from_array(X, 5)
        y2 = da.from_array(np.array([0, 1, 2, 0, 1, 2, 0, 0, 0, 1]), chunks=5)
    else:
        X2 = dd.from_pandas(df, npartitions=2)
        y2 = dd.from_pandas(labels, npartitions=2)

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            a = dxgb.XGBClassifier(
                num_class=3, n_estimators=10, objective="multi:softprob"
            )
            a.fit(X2, y2)
            p1 = a.predict(X2)

            assert dask.is_dask_collection(p1)

            if kind == "array":
                assert p1.shape == (10,)

            result = p1.compute()
            assert result.shape == (10,)

            # proba
            p2 = a.predict_proba(X2)
            assert dask.is_dask_collection(p2)

            if kind == "array":
                assert p2.shape == (10, 3)
            assert p2.compute().shape == (10, 3)


def test_regressor(loop):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            a = dxgb.XGBRegressor()
            X2 = da.from_array(X, 5)
            y2 = da.from_array(y, 5)
            a.fit(X2, y2)
            p1 = a.predict(X2)

    b = xgb.XGBRegressor()
    b.fit(X, y)
    assert_eq(p1, b.predict(X))


def test_regressor_with_early_stopping(loop):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            a = dxgb.XGBRegressor()
            X2 = da.from_array(X, 5)
            y2 = da.from_array(y, 5)
            a.fit(
                X2, y2, early_stopping_rounds=4, eval_metric="rmse", eval_set=[(X, y)],
            )
            p1 = a.predict(X2)

    b = xgb.XGBRegressor()
    b.fit(X, y, early_stopping_rounds=4, eval_metric="rmse", eval_set=[(X, y)])
    assert_eq(p1, b.predict(X))
    assert_eq(a.best_score, b.best_score)


def test_validation_weights_xgbregressor(loop):  # noqa
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error

    # prepare training and test data
    X, y = make_regression(n_samples=2000, random_state=42)

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            X_train, X_test = X[:1600], X[1600:]
            y_train, y_test = y[:1600], y[1600:]

            dX_train = da.from_array(X_train)
            dy_train = da.from_array(y_train)
            dX_test = da.from_array(X_test)

            reg = dxgb.XGBRegressor()

            reg.fit(
                dX_train, dy_train,  # sample_weight=weights_train,
            )
            preds = reg.predict(dX_test)

            rng = np.random.RandomState(0)
            weights_train = 100.0 + rng.rand(len(X_train))
            weights_train = da.from_array(weights_train)
            weights_test = 100.0 + rng.rand(len(X_test))

            reg.fit(
                dX_train,
                dy_train,
                sample_weight=weights_train,
                sample_weight_eval_set=[weights_test],
            )
            preds2 = reg.predict(dX_test)

    err = mean_squared_error(preds, y_test)
    err2 = mean_squared_error(preds2, y_test)
    assert err != err2


@gen_cluster(client=True, timeout=None)
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


@gen_cluster(client=True, timeout=None)
def test_dmatrix_kwargs(c, s, a, b):
    xgb.rabit.init()  # workaround for "Doing rabit call after Finalize"
    dX = da.from_array(X, chunks=(2, 2))
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb.train(c, param, dX, dy, dmatrix_kwargs={"missing": 0.0})

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


@gen_cluster(client=True, timeout=None)
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


@gen_cluster(client=True, timeout=None)
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


@gen_cluster(client=True, timeout=None)
def test_sparse(c, s, a, b):
    xgb.rabit.init()  # workaround for "Doing rabit call after Finalize"
    dX = da.from_array(X, chunks=(2, 2)).map_blocks(scipy.sparse.csr_matrix)
    dy = da.from_array(y, chunks=(2,))
    dbst = yield dxgb.train(c, param, dX, dy)
    dbst = yield dxgb.train(c, param, dX, dy)  # we can do this twice

    predictions = dxgb.predict(c, dbst, dX)
    assert isinstance(predictions, da.Array)

    predictions_result = yield c.compute(predictions)
    _test_container(dbst, predictions_result, scipy.sparse.csr_matrix)


def test_synchronous_api(loop):  # noqa
    dtrain = xgb.DMatrix(df, label=labels)
    bst = xgb.train(param, dtrain)

    ddf = dd.from_pandas(df, npartitions=4)
    dlabels = dd.from_pandas(labels, npartitions=4)

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:

            dbst = dxgb.train(c, param, ddf, dlabels)

            result = bst.predict(dtrain)
            dresult = dbst.predict(dtrain)

            correct = (result > 0.5) == labels
            dcorrect = (dresult > 0.5) == labels
            assert dcorrect.sum() >= correct.sum()


@gen_cluster(client=True, timeout=None)
def test_errors(c, s, a, b):
    def f(part):
        raise Exception("foo")

    df = dd.demo.make_timeseries()
    df = df.map_partitions(f, meta=df._meta)

    with pytest.raises(Exception) as info:
        yield dxgb.train(c, param, df, df.x)

    assert "foo" in str(info.value)


@gen_cluster(client=True, timeout=None)
@pytest.mark.asyncio
async def test_predict_proba(c, s, a, b):
    X = da.random.random((50, 2), chunks=25)
    y = da.random.randint(0, 2, size=50, chunks=25)
    X_ = await c.compute(X)

    # array
    clf = dxgb.XGBClassifier()
    clf.fit(X, y, classes=[0, 1])
    booster = await clf._Booster

    result = clf.predict_proba(X_)
    expected = booster.predict(xgb.DMatrix(X_))
    np.testing.assert_array_equal(result, expected)

    # dataframe
    XX = dd.from_dask_array(X, columns=["A", "B"])
    yy = dd.from_dask_array(y)
    XX_ = await c.compute(XX)

    clf = dxgb.XGBClassifier()
    clf.fit(XX, yy, classes=[0, 1])
    booster = await clf._Booster

    result = clf.predict_proba(XX_)
    expected = booster.predict(xgb.DMatrix(XX_))
    np.testing.assert_array_equal(result, expected)


def test_regressor_evals_result(loop):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            a = dxgb.XGBRegressor()
            X2 = da.from_array(X, 5)
            y2 = da.from_array(y, 5)
            a.fit(X2, y2, eval_metric="rmse", eval_set=[(X, y)])
            evals_result = a.evals_result()

    b = xgb.XGBRegressor()
    b.fit(X, y, eval_metric="rmse", eval_set=[(X, y)])
    assert_eq(evals_result, b.evals_result())


def test_classifier_evals_result(loop):  # noqa
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            a = dxgb.XGBClassifier()
            X2 = da.from_array(X, 5)
            y2 = da.from_array(y, 5)
            a.fit(X2, y2, eval_metric="rmse", eval_set=[(X, y)])
            evals_result = a.evals_result()

    b = xgb.XGBClassifier()
    b.fit(X, y, eval_metric="rmse", eval_set=[(X, y)])
    assert_eq(evals_result, b.evals_result())


@gen_cluster(client=True, timeout=None)
def test_eval_set_dask_collection_exception(c, s, a, b):
    ddf = dd.from_pandas(df, npartitions=4)
    dlabels = dd.from_pandas(labels, npartitions=4)

    X2 = da.from_array(X, 5)
    y2 = da.from_array(y, 5)

    with pytest.raises(TypeError) as info:
        yield dxgb.train(c, param, ddf, dlabels, eval_set=[(X2, y2)])

    assert "Evaluation set must not contain dask collections." in str(info.value)


@gen_cluster(client=True, timeout=None)
def test_sample_weight_eval_set_dask_collection_exception(c, s, a, b):
    ddf = dd.from_pandas(df, npartitions=4)
    dlabels = dd.from_pandas(labels, npartitions=4)

    X2 = da.from_array(X, 5)
    y2 = da.from_array(y, 5)

    with pytest.raises(TypeError) as info:
        yield dxgb.train(c, param, ddf, dlabels, sample_weight_eval_set=[(X2, y2)])

    assert "Sample weight evaluation set must not contain dask collections." in str(
        info.value
    )
