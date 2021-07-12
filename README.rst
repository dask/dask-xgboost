Dask-XGBoost
============

.. warning::

   Dask-XGBoost has been deprecated and is no longer maintained. The functionality
   of this project has been included directly in XGBoost. To use Dask and XGBoost
   together, please use ``xgboost.dask`` instead
   https://xgboost.readthedocs.io/en/latest/tutorials/dask.html.

Distributed training with XGBoost and Dask.distributed

This repository offers a legacy option to perform distributed training
with XGBoost on Dask.array and Dask.dataframe collections.

::

   pip install dask-xgboost

Please note that XGBoost now includes a Dask API as part of its official Python package.
That API is independent of `dask-xgboost` and is now the recommended way to use Dask
adn XGBoost together. See
`the xgb.dask documentation here https://xgboost.readthedocs.io/en/latest/tutorials/dask.html`
for more details on the new API.



Example
-------

.. code-block:: python

   from dask.distributed import Client
   client = Client('scheduler-address:8786')  # connect to cluster

   import dask.dataframe as dd
   df = dd.read_csv('...')  # use dask.dataframe to load and
   df_train = ...           # preprocess data
   labels_train = ...

   import dask_xgboost as dxgb
   params = {'objective': 'binary:logistic', ...}  # use normal xgboost params
   bst = dxgb.train(client, params, df_train, labels_train)

   >>> bst  # Get back normal XGBoost result
   <xgboost.core.Booster at ... >

   predictions = dxgb.predict(client, bst, data_test)


How this works
--------------

For more information on using Dask.dataframe for preprocessing see the
`Dask.dataframe documentation <http://dask.pydata.org/en/latest/dataframe.html>`_.

Once you have created suitable data and labels we are ready for distributed
training with XGBoost.  Every Dask worker sets up an XGBoost slave and gives
them enough information to find each other.  Then Dask workers hand their
in-memory Pandas dataframes to XGBoost (one Dask dataframe is just many Pandas
dataframes spread around the memory of many machines).  XGBoost handles
distributed training on its own without Dask interference.  XGBoost then hands
back a single ``xgboost.Booster`` result object.


Larger Example
--------------

For a more serious example see

-  `This blogpost <http://matthewrocklin.com/blog/work/2017/03/28/dask-xgboost>`_
-  `This notebook <https://gist.github.com/mrocklin/19c89d78e34437e061876a9872f4d2df>`_
-  `This screencast <https://youtu.be/Cc4E-PdDSro>`_

History
-------

Conversation during development happened at `dmlc/xgboost #2032
<https://github.com/dmlc/xgboost/issues/2032>`_
