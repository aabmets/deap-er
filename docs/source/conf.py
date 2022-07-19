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


# -- Project information -----------------------------------------------------
project = 'deap-er'
copyright = '2022, Mattias Aabmets'
author = 'Mattias Aabmets'
release = '0.4.0'


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc'
]
exclude_patterns = []
templates_path = ['_templates']


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
highlight_language = 'python3'
pygments_style = 'default'
html_static_path = ['_static']


# -- Options for Autodoc -------------------------------------------------
autodoc_mock_imports = [
    'numpy',
    'ray'
]
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = True
autodoc_warningiserror = True
autodoc_inherit_docstrings = True
autoclass_content = "both"
autodoc_class_signature = 'mixed'
autodoc_member_order = 'bysource'
autodoc_default_options = {}
autodoc_docstring_signature = True
autodoc_type_aliases = {}


# -- Options for Napoleon -------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_attr_annotations = True
napoleon_type_aliases = None
