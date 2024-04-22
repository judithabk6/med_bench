import os
import sys

import med_bench
import med_bench.get_estimation
import med_bench.get_simulated_data
import med_bench.mediation
import med_bench.utils
import med_bench.utils.utils
import med_bench.utils.nuisances


sys.path.insert(0, os.path.abspath('../'))

project = 'med_bench'
copyright = '2024, Judith Abecassis, Houssam Zenati, Bertrand Thirion, Hadrien Mariaccia, Mouad Zbakh'
author = 'Judith Abecassis, Houssam Zenati, Bertrand Thirion, Hadrien Mariaccia, Mouad Zbakh'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
