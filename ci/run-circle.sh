#!/usr/bin/env bash

echo "[running tests]"
export PATH="$MINICONDA_DIR/bin:$PATH"

source activate pandas

echo "pytest dask_xgboost"
pytest dask_xgboost
