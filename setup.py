#!/usr/bin/env python

import os
from setuptools import setup

requires = open("requirements.txt").read().strip().split("\n")
install_requires = []
extras_require = {"sparse": ["sparse", "scipy"]}
for r in requires:
    if ";" in r:
        # requirements.txt conditional dependencies need to be reformatted for
        # wheels to the form: `'[extra_name]:condition' : ['requirements']`
        req, cond = r.split(";", 1)
        cond = ":" + cond
        cond_reqs = extras_require.setdefault(cond, [])
        cond_reqs.append(req)
    else:
        install_requires.append(r)

setup(
    name="dask-xgboost",
    version="0.1.11",
    description="Interactions between Dask and XGBoost",
    maintainer="Matthew Rocklin",
    maintainer_email="mrocklin@continuum.io",
    url="https://github.com/dask/dask-xgboost",
    license="BSD",
    install_requires=install_requires,
    extras_require=extras_require,
    packages=["dask_xgboost"],
    long_description=(
        open("README.rst").read() if os.path.exists("README.rst") else ""
    ),
    zip_safe=False,
)
