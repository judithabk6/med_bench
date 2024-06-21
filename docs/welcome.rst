Getting Started
===================================


`med_bench` is an open-source Python software package for mediation analysis
task. The goal of `med_bench` is to provide benchmark estimators
in mediation models, based on a comprehensive study of the causal inference
litterature. `med_bench` can handle binary treatments, binary, continuous and
multi-dimensional mediators and provides a simple Python API.

The code is originally written by Judith Abecassis (Inria, Saclay, Soda team).


Licence 
*******


`med_bench` is distributed under BSD-3-Clause license.


Installation
**************

`med_bench` can be installed by executing::

    python setup.py install


Or the package can be directly installed from the GitHub repository using::

    pip install git+git://github.com/judithabk6/med_bench.git


Installation time is a few minutes on a standard personal computer.

Some estimators rely on their R implementation which requires the installation of the corresponding R packages. This can be done using `rpy2`::

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

.. image:: logos/inria_logo.png
   :width: 30%
.. image:: logos/logo_soda.png
   :width: 30%
.. image:: logos/logo_mind.png
   :width: 20%