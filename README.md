# Rule Covering for Interpretation and Boosting

We propose two mathematical programming based algorithms for
interpretation and boosting of tree-based ensemble methods. These
algorithms are called minimum rule cover (MIRCO) and rule cover
boosting (RCBoost). The details of both algorithms are given in our
[paper](https://arxiv.org/abs/2007.06379). In this note, we introduce our implementation of both
algorithms as well as list the steps to reproduce our results.

## Required packages

All our codes are implemented in Python 3.7 and we use the following
packages:

1. [`scikit-learn`](https://scikit-learn.org/stable/index.html)
2. [`numpy`](https://numpy.org/)
3. [`gurobipy`](https://pypi.org/project/gurobipy/)

We have used the standard installation of [Anaconda
Distribution](https://www.anaconda.com/products/individual) (Pyhton
3.7), with which the first two packages are already bundled. The third
packages can be separately installed again by the Anaconda package
manager. Note that along with the Python package,you also need to
install [Gurobi
Optimizer](https://www.gurobi.com/academia/academic-program-and-licenses/),
which is free for research and educational work.

## Tutorials

In order to test MIRCO on a set of test problems, we refer to page
[`MIRCO.html`](MIRCO.html) or to the [notebook](MIRCO.ipynb). Likewise,
for RCBoost we have prepared another page
[`RCBoost.html`](RCBoost.html) and a [notebook](RCBoost.ipynb).

## Reproducing our results

We provide two scripts [`MIRCO_run.py`](MIRCO_run.py) and
[`RCBoost_run.py`](RCBoost_run.py). Running these scripts should
reproduce the results that we have reported in our [paper](https://arxiv.org/abs/2007.06379).
