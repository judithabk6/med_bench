import os
import sys

import med_bench
import med_bench.get_simulated_data
import med_bench.estimation
from med_bench.estimation.mediation_coefficient_product import CoefficientProduct
from med_bench.estimation.mediation_dml import DoubleMachineLearning
from med_bench.estimation.mediation_g_computation import GComputation
from med_bench.estimation.mediation_ipw import InversePropensityWeighting
from med_bench.estimation.mediation_mr import MultiplyRobust
from med_bench.estimation.mediation_tmle import TMLE
import med_bench.utils
import med_bench.utils.utils


sys.path.insert(0, os.path.abspath('../'))

project = 'med_bench'
copyright = '2025, Judith Abecassis, Houssam Zenati, Bertrand Thirion, Hadrien Mariaccia, Mouad Zbakh, Sami Boumaïza, Julie Josse'
author = 'Judith Abecassis, Houssam Zenati, Bertrand Thirion, Hadrien Mariaccia, Mouad Zbakh, Sami Boumaïza, Julie Josse'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv']

sphinx_gallery_conf = {
    "doc_module": "med_bench",
    'examples_dirs': 'examples',
    'gallery_dirs': 'auto_examples',
    "filename_pattern": ".*",
}
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
