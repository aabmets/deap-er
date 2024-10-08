# Configuration file for the Sphinx documentation builder.
#
# Link to the configuration documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Paths setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../'))
sys.path.append(os.path.abspath("./_static"))


# -- Project information -----------------------------------------------------
project = 'deap-er'
copyright = '2022, Mattias Aabmets'
author = 'Mattias Aabmets'
release = '2.0.0'


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.intersphinx',
    'sphinx.ext.autodoc',
    'sphinx_rtd_theme',
    'sphinx_design'
]
exclude_patterns = []
templates_path = ['_templates']
add_module_names = False


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
pygments_style = 'custom_style.CustomStyle'
highlight_language = 'python3'
html_static_path = ['_static']


# -- Options for intersphinx -------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None)
}


# -- Options for Autodoc -------------------------------------------------
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_typehints_description_target = 'documented'
autodoc_preserve_defaults = True
autodoc_warningiserror = True
autodoc_inherit_docstrings = True
autodoc_docstring_signature = True
autoclass_content = "both"
autodoc_class_signature = 'mixed'
autodoc_member_order = 'bysource'
autodoc_default_options = {}
autodoc_type_aliases = {}
autodoc_mock_imports = ['numpy', 'scipy', 'dill']
