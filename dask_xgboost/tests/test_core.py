import numpy as np
import xgboost as xgb

from distributed import Nanny, Client
from distributed.utils_test import gen_cluster, loop, cluster

import dask_xgboost as dxgb

def test_simple():
    x = np.random.random((1000, 10))
    y = np.random.random(1000)
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}


@gen_cluster(client=True, Worker=Nanny)
def test_dask(c, s, a, b):
    x = np.random.random((1000, 10))
    y = np.random.random(1000)

    dtrain = xgb.DMatrix(x, label=y)

    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}

    bst = xgb.train(param, dtrain)

    dbst = yield dxgb._train(c, param, x, y)

    assert bst.get_dump() == dbst# .get_dump()


def test_synchronous_api(loop):
    param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}

    x = np.random.random((1000, 10))
    y = np.random.random(1000)

    dtrain = xgb.DMatrix(x, label=y)
    bst = xgb.train(param, dtrain)

    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:

            dbst = dxgb.train(c, param, x, y)
            assert bst.get_dump() == dbst# .get_dump()
