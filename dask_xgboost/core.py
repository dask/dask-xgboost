from collections import defaultdict
import logging
from threading import Thread

import dask.dataframe as dd
import dask.array as da
import numpy as np
import pandas as pd
from toolz import first, assoc
from tornado import gen
from dask import delayed
from distributed import Client
from distributed.client import _wait, default_client
from distributed.utils import sync
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
    else:
        raise TypeError("Data must be either numpy arrays or pandas dataframes"
                        ". Got %s" % type(L[0]))


def train_part(env, param, list_of_parts, **kwargs):
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
    feature_names = getattr(data, 'columns', None)
    dtrain = xgb.DMatrix(data, labels, feature_names=feature_names)

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
def _train(client, params, data, labels, **kwargs):
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
    yield _wait(parts)

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
                             assoc(params, 'nthreads', ncores[worker]),
                             list_of_parts, workers=worker, **kwargs)
               for worker, list_of_parts in worker_map.items()]

    # Get the results, only one will be non-None
    results = yield client._gather(futures)
    result = [v for v in results if v][0]
    raise gen.Return(result)


def train(client, params, data, labels, **kwargs):
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
    **kwargs:
        Keywords to give to XGBoost

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
    return sync(client.loop, _train, client, params, data, labels, **kwargs)


def _predict_part(part, model=None):
    xgb.rabit.init()
    dm = xgb.DMatrix(part)
    result = model.predict(dm)
    xgb.rabit.finalize()
    if isinstance(part, pd.DataFrame):
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
    elif isinstance(data, da.Array):
        result = data.map_blocks(_predict_part, model=model, dtype=float,
                                 drop_axis=1)

    return result


class XGBRegressor:
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective='reg:linear', nthread=-1, gamma=0,
                 min_child_weight=1, max_delta_step=0, subsample=1,
                 colsample_bytree=1, colsample_bylevel=1, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0,
                 missing=None, scheduler_address=None):
        self.scheduler_address = scheduler_address
        super().__init__(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            silent=silent,
            objective=objective,
            nthread=nthread,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            seed=seed,
            missing=missing,
            scheduler_address=scheduler_address)

    def fit(self, X, y=None):
        client = Client(self.scheduler_address)
        params = self.get_params()
        client = Client(params.pop('scheduler_address'))

        train(client, params, X, y)


class XGBClassifier(xgb.XGBClassifier):
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective='binary:logistic', nthread=-1, gamma=0,
                 min_child_weight=1, max_delta_step=0, subsample=1,
                 colsample_bytree=1, colsample_bylevel=1, reg_alpha=0,
                 reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0,
                 missing=None):
        super().__init__(
            base_score=base_score,
            colsample_bylevel=colsample_bylevel,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            learning_rate=learning_rate,
            max_delta_step=max_delta_step,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            missing=missing,
            n_estimators=n_estimators,
            nthread=nthread,
            objective=objective,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            seed=seed,
            silent=silent)

    def fit(self, X, y=None):
        client = default_client()

        xgb_options = self.get_xgb_params()
        self._Booster = train(client, xgb_options, X, y,
                              num_boost_round=self.n_estimators)
        return self

    def predict(self, X):
        client = default_client()
        return predict(client, self._Booster, X)
