from collections import defaultdict
import logging
from threading import Thread

import numpy as np
import pandas as pd
from toolz import first, assoc
from tornado import gen

try:
    import sparse
    import scipy.sparse as ss
except ImportError:
    sparse = False
    ss = False

from dask import delayed
from dask.distributed import wait, default_client
import dask.dataframe as dd
import dask.array as da

import xgboost as xgb

from .tracker import RabitTracker

logger = logging.getLogger(__name__)


def parse_host_port(address):
    if '://' in address:
        address = address.rsplit('://', 1)[1]
    host, port = address.split(':')
    port = int(port)
    return host, port


def start_tracker(host, n_workers):
    """ Start Rabit tracker """
    env = {'DMLC_NUM_WORKER': n_workers}
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
    elif ss and isinstance(L[0], ss.spmatrix):
        return ss.vstack(L, format='csr')
    elif sparse and isinstance(L[0], sparse.SparseArray):
        return sparse.concatenate(L, axis=0)
    else:
        raise TypeError("Data must be either numpy arrays or pandas dataframes"
                        ". Got %s" % type(L[0]))


def train_part(env, param, list_of_parts, dmatrix_kwargs=None, **kwargs):
    """
    Run part of XGBoost distributed workload

    This starts an xgboost.rabit slave, trains on provided data, and then shuts
    down the xgboost.rabit slave

    Returns
    -------
    model if rank zero, None otherwise
    """
    data, labels = zip(*list_of_parts)  # Prepare data
    data = concat(data)                 # Concatenate many parts into one
    labels = concat(labels)
    if dmatrix_kwargs is None:
        dmatrix_kwargs = {}

    dmatrix_kwargs["feature_names"] = getattr(data, 'columns', None)
    dtrain = xgb.DMatrix(data, labels, **dmatrix_kwargs)

    args = [('%s=%s' % item).encode() for item in env.items()]
    xgb.rabit.init(args)
    try:
        logger.info("Starting Rabit, Rank %d", xgb.rabit.get_rank())

        bst = xgb.train(param, dtrain, **kwargs)

        if xgb.rabit.get_rank() == 0:  # Only return from one worker
            result = bst
        else:
            result = None
    finally:
        xgb.rabit.finalize()
    return result


@gen.coroutine
def _train(client, params, data, labels, dmatrix_kwargs={}, **kwargs):
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

    # Arrange parts into pairs.  This enforces co-locality
    parts = list(map(delayed, zip(data_parts, label_parts)))
    parts = client.compute(parts)  # Start computation in the background
    yield wait(parts)

    for part in parts:
        if part.status == 'error':
            yield part  # trigger error locally

    # Because XGBoost-python doesn't yet allow iterative training, we need to
    # find the locations of all chunks and map them to particular Dask workers
    key_to_part_dict = dict([(part.key, part) for part in parts])
    who_has = yield client.scheduler.who_has(keys=[part.key for part in parts])
    worker_map = defaultdict(list)
    for key, workers in who_has.items():
        worker_map[first(workers)].append(key_to_part_dict[key])

    ncores = yield client.scheduler.ncores()  # Number of cores per worker

    # Start the XGBoost tracker on the Dask scheduler
    host, port = parse_host_port(client.scheduler.address)
    env = yield client._run_on_scheduler(start_tracker,
                                         host.strip('/:'),
                                         len(worker_map))

    # Tell each worker to train on the chunks/parts that it has locally
    futures = [client.submit(train_part, env,
                             assoc(params, 'nthread', ncores[worker]),
                             list_of_parts, workers=worker,
                             dmatrix_kwargs=dmatrix_kwargs, **kwargs)
               for worker, list_of_parts in worker_map.items()]

    # Get the results, only one will be non-None
    results = yield client._gather(futures)
    result = [v for v in results if v][0]
    num_class = params.get("num_class")
    if num_class:
        result.set_attr(num_class=str(num_class))
    raise gen.Return(result)


def train(client, params, data, labels, dmatrix_kwargs={}, **kwargs):
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
    return client.sync(_train, client, params, data,
                       labels, dmatrix_kwargs, **kwargs)


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
            result = pd.Series(result, index=part.index, name='predictions')
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
            kwargs = dict(
                drop_axis=None,
                chunks=(data.chunks[0], (num_class,)),
            )
        else:
            kwargs = dict(
                drop_axis=1,
            )
        result = data.map_blocks(_predict_part, model=model,
                                 dtype=np.float32,
                                 **kwargs)

    return result


class XGBRegressor(xgb.XGBRegressor):

    def fit(self, X, y=None):
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
        """
        client = default_client()
        xgb_options = self.get_xgb_params()
        self._Booster = train(client, xgb_options, X, y,
                              num_boost_round=self.n_estimators)
        return self

    def predict(self, X):
        client = default_client()
        return predict(client, self._Booster, X)


class XGBClassifier(xgb.XGBClassifier):

    def fit(self, X, y=None, classes=None):
        """Fit a gradient boosting classifier

        Parameters
        ----------
        X : array-like [n_samples, n_features]
            Feature Matrix. May be a dask.array or dask.dataframe
        y : array-like
            Labels
        classes : sequence, optional
            The unique values in `y`. If no specified, this will be
            eagerly computed from `y` before training.

        Returns
        -------
        self : XGBClassifier

        Notes
        -----
        This differs from the XGBoost version in three ways

        1. The ``sample_weight``, ``eval_set``, ``eval_metric``,
          ``early_stopping_rounds`` and ``verbose`` fit kwargs are not
          supported.
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

        if self.n_classes_ > 2:
            # xgboost just ignores the user-provided objective
            # We only overwrite if it's the default...
            if xgb_options['objective'] == "binary:logistic":
                xgb_options["objective"] = "multi:softprob"

            xgb_options.setdefault('num_class', self.n_classes_)

        # xgboost sets this to self.objective, which I think is wrong
        # hyper-parameters should not be updated during fit.
        self.objective = xgb_options['objective']

        # TODO: auto label-encode y
        # that will require a dependency on dask-ml
        # TODO: sample weight

        self._Booster = train(client, xgb_options, X, y,
                              num_boost_round=self.n_estimators)
        return self

    def predict(self, X):
        client = default_client()
        class_probs = predict(client, self._Booster, X)
        if class_probs.ndim > 1:
            cidx = da.argmax(class_probs, axis=1)
        else:
            cidx = (class_probs > 0).astype(np.int64)
        return cidx

    def predict_proba(self, data, ntree_limit=None):
        client = default_client()
        if ntree_limit is not None:
            raise NotImplementedError("'ntree_limit' is not currently "
                                      "supported.")
        class_probs = predict(client, self._Booster, data)
        return class_probs
