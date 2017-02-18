import io
import logging
from threading import Thread

from tornado import gen
from distributed.utils import sync
from distributed.comm.core import parse_host_port
import xgboost as xgb

from .tracker import RabitTracker

logger = logging.getLogger(__name__)


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


def worker(env, param, data, labels, **kwargs):
    """
    Run part of XGBoost distributed workload

    This starts an xgboost.rabit slave, trains on provided data, and then shuts
    down the xgboost.rabit slave

    Returns
    -------
    model if rank zero, None otherwise
    """
    dtrain = xgb.DMatrix(data, labels)

    import os # TODO: use these keywords in init directly
              # Queestion: what are the right keywords?
    env = {k: str(v) for k, v in env.items()}
    os.environ.update(env)

    xgb.rabit.init()
    logger.info("Starting Rabit, Rank %d", xgb.rabit.get_rank())

    bst = xgb.train(param, dtrain, **kwargs)

    if xgb.rabit.get_rank() == 0:
        result = bst.get_dump() # TODO: return serializable model directly
                                #       or else rebuild on other side
    else:
        result = None
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
    # TODO: we would like to be able to send DMatrix objects directly
    # TODO: handle distributed data
    cluster_info = yield client.scheduler.identity()
    host, port = parse_host_port(client.scheduler.address)
    env = yield client._run_on_scheduler(start_tracker,
                                         host.strip('/:'),
                                         len(cluster_info['workers']))
    result = yield client._run(worker, env, params, data, labels, **kwargs)
    result = [v for v in result.values() if v]
    return result


def train(client, params, data, labels, **kwargs):
    """ Train an XGBoost model on a Dask Cluster

    This starts XGBoost on all Dask workers, moves input data to those workers,
    and then calls ``xgboost.train`` on the inputs.

    Examples
    --------

    >>> client = Client('scheduler-address:8786')  # doctest: +SKIP
    >>> train(client, params, data, labels, **normal_kwargs)  # doctest: +SKIP
    """
    return sync(client.loop, _train, client, params, data, labels, **kwargs)


"""
TODO
====

-   Serialize DMatrix objects
-   Serialize Booster objects
-   Pass keywords directly to rabit.init rather than use environment variables
    (current approach fails if multiple workers are in the same process)
-   Move initial data to workers more efficiently through existing tree broadcast
-   Support proper dask.arrays/dataframes and distributed data
"""
