import warnings

from .core import XGBClassifier, XGBRegressor, _train, predict, train  # noqa

__version__ = "0.2.0"

warnings.warn(
   "Dask-XGBoost has been deprecated and is no longer maintained. The functionality "
   "of this project has been included directly in XGBoost. To use Dask and XGBoost "
   "together, please use ``xgboost.dask`` instead "
   "https://xgboost.readthedocs.io/en/latest/tutorials/dask.html."
)