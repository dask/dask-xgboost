import logging
from collections import defaultdict
from threading import Thread

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import xgboost as xgb
from dask import delayed, is_dask_collection
from dask.distributed import default_client, wait
from toolz import assoc, first
from tornado import gen

from .tracker import RabitTracker, get_host_ip

try:
    import sparse
except ImportError:
    sparse = False

try:
    import scipy.sparse as ss
except ImportError:
    ss = False

logger = logging.getLogger(__name__)


def parse_host_port(address):
    if "://" in address:
        address = address.rsplit("://", 1)[1]
    host, port = address.split(":")
    port = int(port)
    return host, port


def start_tracker(host, n_workers):
    """ Start Rabit tracker """
    if host is None:
        host = get_host_ip("auto")
    env = {"DMLC_NUM_WORKER": n_workers}
    rabit = RabitTracker(hostIP=host, nslave=n_workers)
    env.update(rabit.slave_envs())

    rabit.start(n_workers)
    logger.info("Starting Rabit Tracker")
    thread = Thread(target=rabit.join)
    thread.daemon = True
    thread.start()
    return env


def concat(L):
    if isinstance(L[0], np.ndarray):
        return np.concatenate(L, axis=0)
    elif isinstance(L[0], (pd.DataFrame, pd.Series)):
        return pd.concat(L, axis=0)
    elif ss and isinstance(L[0], ss.csr_matrix):
        return ss.vstack(L, format="csr")
    elif sparse and isinstance(L[0], sparse.SparseArray):
        return sparse.concatenate(L, axis=0)
    else:
        raise TypeError(
            "Data must be either numpy arrays or pandas dataframes"
            ". Got %s" % type(L[0])
        )


def train_part(
    env,
    param,
    list_of_parts,
    dmatrix_kwargs=None,
    eval_set=None,
    missing=None,
    n_jobs=None,
    sample_weight_eval_set=None,
    **kwargs
):
    """
    Run part of XGBoost distributed workload

    This starts an xgboost.rabit slave, trains on provided data, and then shuts
    down the xgboost.rabit slave

    Returns
    -------
    model if rank zero, None otherwise
    """
    data, labels, sample_weight = zip(*list_of_parts)  # Prepare data
    data = concat(data)  # Concatenate many parts into one
    labels = concat(labels)
    sample_weight = concat(sample_weight) if np.all(sample_weight) else None

    if dmatrix_kwargs is None:
        dmatrix_kwargs = {}

    dmatrix_kwargs["feature_names"] = getattr(data, "columns", None)
    dtrain = xgb.DMatrix(data, labels, weight=sample_weight, **dmatrix_kwargs)

    evals = _package_evals(
        eval_set,
        sample_weight_eval_set=sample_weight_eval_set,
        missing=missing,
        n_jobs=n_jobs,
    )

    args = [("%s=%s" % item).encode() for item in env.items()]
    xgb.rabit.init(args)
    try:
        local_history = {}
        logger.info("Starting Rabit, Rank %d", xgb.rabit.get_rank())
        bst = xgb.train(
            param, dtrain, evals=evals, evals_result=local_history, **kwargs
        )

        if xgb.rabit.get_rank() == 0:  # Only return from one worker
            result = bst
            evals_result = local_history
        else:
            result = None
            evals_result = None
    finally:
        logger.info("Finalizing Rabit, Rank %d", xgb.rabit.get_rank())
        xgb.rabit.finalize()
    return result, evals_result


def _package_evals(eval_set, sample_weight_eval_set=None, missing=None, n_jobs=None):
    if eval_set is not None:
        if sample_weight_eval_set is None:
            sample_weight_eval_set = [None] * len(eval_set)
        evals = list(
            xgb.DMatrix(
                data, label=label, missing=missing, weight=weight, nthread=n_jobs,
            )
            for ((data, label), weight) in zip(eval_set, sample_weight_eval_set)
        )
        evals = list(zip(evals, ["validation_{}".format(i) for i in range(len(evals))]))
    else:
        evals = ()
    return evals


def _has_dask_collections(list_of_collections, message):
    list_of_collections = list_of_collections or []
    if any(
        is_dask_collection(collection)
        for collections in list_of_collections
        for collection in collections
    ):
        raise TypeError(message)


@gen.coroutine
def _train(
    client,
    params,
    data,
    labels,
    dmatrix_kwargs={},
    evals_result=None,
    sample_weight=None,
    **kwargs
):
    """
    Asynchronous version of train

    See Also
    --------
    train
    """
    # Break apart Dask.array/dataframe into chunks/parts
    data_parts = data.to_delayed()
    label_parts = labels.to_delayed()
    if isinstance(data_parts, np.ndarray):
        assert data_parts.shape[1] == 1
        data_parts = data_parts.flatten().tolist()
    if isinstance(label_parts, np.ndarray):
        assert label_parts.ndim == 1 or label_parts.shape[1] == 1
        label_parts = label_parts.flatten().tolist()
    if sample_weight is not None:
        sample_weight_parts = sample_weight.to_delayed()
        if isinstance(sample_weight_parts, np.ndarray):
            assert sample_weight_parts.ndim == 1 or sample_weight_parts.shape[1] == 1
            sample_weight_parts = sample_weight_parts.flatten().tolist()
    else:
        # If sample_weight is None construct a list of Nones to keep
        # the structure of parts consistent.
        sample_weight_parts = [None] * len(data_parts)

    # Check that data, labels, and sample_weights are the same length
    lists = [data_parts, label_parts, sample_weight_parts]
    if len(set([len(l) for l in lists])) > 1:
        raise ValueError(
            "data, label, and sample_weight parts/chunks must have same length."
        )

    # Arrange parts into triads.  This enforces co-locality
    parts = list(map(delayed, zip(data_parts, label_parts, sample_weight_parts)))
    parts = client.compute(parts)  # Start computation in the background
    yield wait(parts)

    for part in parts:
        if part.status == "error":
            yield part  # trigger error locally

    _has_dask_collections(
        kwargs.get("eval_set", []), "Evaluation set must not contain dask collections."
    )
    _has_dask_collections(
        kwargs.get("sample_weight_eval_set", []),
        "Sample weight evaluation set must not contain dask collections.",
    )

    # Because XGBoost-python doesn't yet allow iterative training, we need to
    # find the locations of all chunks and map them to particular Dask workers
    key_to_part_dict = dict([(part.key, part) for part in parts])
    who_has = yield client.scheduler.who_has(keys=[part.key for part in parts])
    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[first(workers)].append(key_to_part_dict[key])

    ncores = yield client.scheduler.ncores()  # Number of cores per worker

    # Start the XGBoost tracker on the Dask scheduler
    env = yield client._run_on_scheduler(start_tracker, None, len(worker_map))

    # Tell each worker to train on the chunks/parts that it has locally
    futures = [
        client.submit(
            train_part,
            env,
            assoc(params, "nthread", ncores[worker]),
            list_of_parts,
            workers=worker,
            dmatrix_kwargs=dmatrix_kwargs,
            **kwargs
        )
        for worker, list_of_parts in worker_map.items()
    ]

    # Get the results, only one will be non-None
    results = yield client._gather(futures)
    result, _evals_result = [v for v in results if v.count(None) != len(v)][0]

    if evals_result is not None:
        evals_result.update(_evals_result)

    num_class = params.get("num_class")
    if num_class:
        result.set_attr(num_class=str(num_class))
    raise gen.Return(result)


def train(
    client,
    params,
    data,
    labels,
    dmatrix_kwargs={},
    evals_result=None,
    sample_weight=None,
    **kwargs
):
    """ Train an XGBoost model on a Dask Cluster

    This starts XGBoost on all Dask workers, moves input data to those workers,
    and then calls ``xgboost.train`` on the inputs.

    Parameters
    ----------
    client: dask.distributed.Client
    params: dict
        Parameters to give to XGBoost (see xgb.Booster.train)
    data: dask array or dask.dataframe
    labels: dask.array or dask.dataframe
    dmatrix_kwargs: Keywords to give to Xgboost DMatrix
    evals_result: dict, optional
        Stores the evaluation result history of all the items in the eval_set
        by mutating evals_result in place.
    sample_weight : array_like, optional
        instance weights
    **kwargs: Keywords to give to XGBoost train

    Examples
    --------
    >>> client = Client('scheduler-address:8786')  # doctest: +SKIP
    >>> data = dd.read_csv('s3://...')  # doctest: +SKIP
    >>> labels = data['outcome']  # doctest: +SKIP
    >>> del data['outcome']  # doctest: +SKIP
    >>> train(client, params, data, labels, **normal_kwargs)  # doctest: +SKIP
    <xgboost.core.Booster object at ...>

    See Also
    --------
    predict
    """
    return client.sync(
        _train,
        client,
        params,
        data,
        labels,
        dmatrix_kwargs,
        evals_result,
        sample_weight,
        **kwargs
    )


def _predict_part(part, model=None):
    xgb.rabit.init()
    try:
        dm = xgb.DMatrix(part)
        result = model.predict(dm)
    finally:
        xgb.rabit.finalize()

    if isinstance(part, pd.DataFrame):
        if model.attr("num_class"):
            result = pd.DataFrame(result, index=part.index)
        else:
            result = pd.Series(result, index=part.index, name="predictions")
    return result


def predict(client, model, data):
    """ Distributed prediction with XGBoost

    Parameters
    ----------
    client: dask.distributed.Client
    model: xgboost.Booster
    data: dask array or dataframe

    Examples
    --------
    >>> client = Client('scheduler-address:8786')  # doctest: +SKIP
    >>> test_data = dd.read_csv('s3://...')  # doctest: +SKIP
    >>> model
    <xgboost.core.Booster object at ...>

    >>> predictions = predict(client, model, test_data)  # doctest: +SKIP

    Returns
    -------
    Dask.dataframe or dask.array, depending on the input data type

    See Also
    --------
    train
    """
    if isinstance(data, dd._Frame):
        result = data.map_partitions(_predict_part, model=model)
        result = result.values
    elif isinstance(data, da.Array):
        num_class = model.attr("num_class") or 2
        num_class = int(num_class)

        if num_class > 2:
            kwargs = dict(drop_axis=None, chunks=(data.chunks[0], (num_class,)))
        else:
            kwargs = dict(drop_axis=1)
        result = data.map_blocks(_predict_part, model=model, dtype=np.float32, **kwargs)
    else:
        model = model.result()  # Future to concrete
        if not isinstance(data, xgb.DMatrix):
            data = xgb.DMatrix(data)
        result = model.predict(data)

    return result


class XGBRegressor(xgb.XGBRegressor):
    def fit(
        self,
        X,
        y=None,
        eval_set=None,
        sample_weight=None,
        sample_weight_eval_set=None,
        eval_metric=None,
        early_stopping_rounds=None,
    ):
        """Fit the gradient boosting model

        Parameters
        ----------
        X : array-like [n_samples, n_features]
        y : array-like

        Returns
        -------
        self : the fitted Regressor

        Notes
        -----
        This differs from the XGBoost version not supporting the ``eval_set``,
        ``eval_metric``, ``early_stopping_rounds`` and ``verbose`` fit
        kwargs.
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        sample_weight : array_like, optional
            instance weights
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is a list
            of instance weights on the i-th validation set.
        eval_metric : str, list of str, or callable, optional
            If a str, should be a built-in evaluation metric to use. See
            `doc/parameter.rst <https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst>`_.  # noqa: E501
            If a list of str, should be the list of multiple built-in
            evaluation metrics to use.
            If callable, a custom evaluation metric. The call
            signature is ``func(y_predicted, y_true)`` where ``y_true`` will
            be a DMatrix object such that you may need to call the
            ``get_label`` method. It must return a str, value pair where
            the str is a name for the evaluation and value is the value of
            the evaluation function. The callable custom objective is always
            minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at
            least once in every **early_stopping_rounds** round(s) to continue
            training.
            Requires at least one item in **eval_set**.
            The method returns the model from the last iteration (not the best
            one).
            If there's more than one item in **eval_set**, the last entry will
            be used for early stopping.
            If there's more than one metric in **eval_metric**, the last
            metric will be used for early stopping.
            If early stopping occurs, the model will have three additional
            fields:
            ``clf.best_score``, ``clf.best_iteration`` and
            ``clf.best_ntree_limit``.
        """
        client = default_client()
        xgb_options = self.get_xgb_params()

        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                xgb_options.update({"eval_metric": eval_metric})

        self.evals_result_ = {}
        self._Booster = train(
            client,
            xgb_options,
            X,
            y,
            num_boost_round=self.n_estimators,
            eval_set=eval_set,
            sample_weight=sample_weight,
            sample_weight_eval_set=sample_weight_eval_set,
            missing=self.missing,
            n_jobs=self.n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=self.evals_result_,
        )

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit
        return self

    def predict(self, X):
        client = default_client()
        return predict(client, self._Booster, X)


class XGBClassifier(xgb.XGBClassifier):
    def fit(
        self,
        X,
        y=None,
        classes=None,
        eval_set=None,
        sample_weight=None,
        sample_weight_eval_set=None,
        eval_metric=None,
        early_stopping_rounds=None,
    ):
        """Fit a gradient boosting classifier

        Parameters
        ----------
        X : array-like [n_samples, n_features]
            Feature Matrix. May be a dask.array or dask.dataframe
        y : array-like
            Labels
        eval_set : list, optional
            A list of (X, y) tuple pairs to use as validation sets, for which
            metrics will be computed.
            Validation metrics will help us track the performance of the model.
        sample_weight : array_like, optional
            instance weights
        sample_weight_eval_set : list, optional
            A list of the form [L_1, L_2, ..., L_n], where each L_i is a list
            of instance weights on the i-th validation set.
        eval_metric : str, list of str, or callable, optional
            If a str, should be a built-in evaluation metric to use. See
            `doc/parameter.rst <https://github.com/dmlc/xgboost/blob/master/doc/parameter.rst>`_.  # noqa: E501
            If a list of str, should be the list of multiple built-in
            evaluation metrics to use.
            If callable, a custom evaluation metric. The call
            signature is ``func(y_predicted, y_true)`` where ``y_true`` will
            be a DMatrix object such that you may need to call the
            ``get_label`` method. It must return a str, value pair where
            the str is a name for the evaluation and value is the value of
            the evaluation function. The callable custom objective is always
            minimized.
        early_stopping_rounds : int
            Activates early stopping. Validation metric needs to improve at
            least once in every **early_stopping_rounds** round(s) to continue
            training.
            Requires at least one item in **eval_set**.
            The method returns the model from the last iteration (not the best
            one).
            If there's more than one item in **eval_set**, the last entry will
            be used for early stopping.
            If there's more than one metric in **eval_metric**, the last
            metric will be used for early stopping.
            If early stopping occurs, the model will have three additional
            fields:
            ``clf.best_score``, ``clf.best_iteration`` and
            ``clf.best_ntree_limit``.
        classes : sequence, optional
            The unique values in `y`. If no specified, this will be
            eagerly computed from `y` before training.

        Returns
        -------
        self : XGBClassifier

        Notes
        -----
        This differs from the XGBoost version in three ways

        1. The ``verbose`` fit kwargs are not supported.
        2. The labels are not automatically label-encoded
        3. The ``classes_`` and ``n_classes_`` attributes are not learned
        """
        client = default_client()

        if classes is None:
            if isinstance(y, da.Array):
                classes = da.unique(y)
            else:
                classes = y.unique()
            classes = classes.compute()
        else:
            classes = np.asarray(classes)
        self.classes_ = classes
        self.n_classes_ = len(self.classes_)

        xgb_options = self.get_xgb_params()

        if eval_metric is not None:
            if callable(eval_metric):
                eval_metric = None
            else:
                xgb_options.update({"eval_metric": eval_metric})

        if self.n_classes_ > 2:
            # xgboost just ignores the user-provided objective
            # We only overwrite if it's the default...
            if xgb_options["objective"] == "binary:logistic":
                xgb_options["objective"] = "multi:softprob"

            xgb_options.setdefault("num_class", self.n_classes_)

        # xgboost sets this to self.objective, which I think is wrong
        # hyper-parameters should not be updated during fit.
        self.objective = xgb_options["objective"]

        # TODO: auto label-encode y
        # that will require a dependency on dask-ml

        self.evals_result_ = {}
        self._Booster = train(
            client,
            xgb_options,
            X,
            y,
            num_boost_round=self.n_estimators,
            eval_set=eval_set,
            sample_weight=sample_weight,
            sample_weight_eval_set=sample_weight_eval_set,
            missing=self.missing,
            n_jobs=self.n_jobs,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=self.evals_result_,
        )

        if early_stopping_rounds is not None:
            self.best_score = self._Booster.best_score
            self.best_iteration = self._Booster.best_iteration
            self.best_ntree_limit = self._Booster.best_ntree_limit
        return self

    def predict(self, X):
        client = default_client()
        class_probs = predict(client, self._Booster, X)
        if class_probs.ndim > 1:
            cidx = da.argmax(class_probs, axis=1)
        else:
            cidx = (class_probs > 0.5).astype(np.int64)
        return cidx

    def predict_proba(self, data, ntree_limit=None):
        client = default_client()
        if ntree_limit is not None:
            raise NotImplementedError("'ntree_limit' is not currently " "supported.")
        class_probs = predict(client, self._Booster, data)
        return class_probs
