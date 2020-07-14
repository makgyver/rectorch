# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#import sphinx_bootstrap_theme
#import sphinxbootstrap4theme
import pytorch_sphinx_theme
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'rectorch'
copyright = '2020, Mirko Polato'
author = 'Mirko Polato'

# The full version, including alpha/beta/rc tags
release = '0.0.1b'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.intersphinx',
              'numpydoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest',
              #'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax']

napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_google_docstring = False

mathjax_path = "https://cdn.mathjax.org/mathjax/2.7-latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"

intersphinx_mapping = {
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'python': ('https://docs.python.org/3', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/dev', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

#highlight_language = 'none'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

html_logo = 'img/logo_400.png'
html_favicon = 'img/favicon.png'

#html_theme_options = {
    # Navigation bar title. (Default: ``project`` value)
    #'navbar_title': " ",
    #'bootswatch_theme': "lumen"
#}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
