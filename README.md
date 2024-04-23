[![codecov](https://codecov.io/gh/judithabk6/med_bench/graph/badge.svg?token=PASB71N41D)](https://codecov.io/gh/judithabk6/med_bench)

# med_bench

**med_bench** is a Python package designed to wrap the most common estimators for causal mediation analysis in a single framework. We additionally allow for some flexibility in the choice of nuisance parameters models.

The simulations and performances evaluations realized here are presented in the following article

Judith Abécassis, Julie Josse and Bertrand Thirion (2022). **Causal mediation analysis with one or multiple mediators: a comparative study.** [pdf](https://judithabk6.github.io/files/article_mediation_benchmark.pdf)

## Installation
med_bench can be installed by executing
```
python setup.py install
```

Or the package can be directly installed from the GitHub repository using
```
pip install git+git://github.com/judithabk6/med_bench.git
```

Installation time is a few minutes on a standard personal computer.

Some estimators rely on their R implementation which requires the installation of the corresponding R packages. This can be done using `rpy2`

````python
import rpy2
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=33)

utils.install_packages('grf')

utils.install_packages('causalweight')

utils.install_packages('mediation')


utils.install_packages('devtools')
devtools = rpackages.importr('devtools')
devtools.install_github('ohines/plmed')
plmed = rpackages.importr('plmed')
````

## Content
The `src` folder contains the main module with the implementation of the different estimators, the `script` folder contains the function used to simulate data and run the experiments, and the `results` folder contains all available results and code to reproduce the figures.
