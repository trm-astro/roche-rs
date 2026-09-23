# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import roche

project = "roche"
copyright = "2026, Alex Brown"
author = "Alex Brown"
release = roche.__version__
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Sphinx built-in extensions
    "sphinx.ext.autodoc",  # Automatic documentation generation from docstrings
    "sphinx.ext.autosummary",  # Generate summary tables for modules/classes/functions
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "numpydoc",  # Parse Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to source code
    # MyST extensions
    # "myst_parser",  # Markdown support (not needed, included with myst_nb)
    "myst_nb",  # MyST Notebooks support
]

templates_path = ["_templates"]  # For custom HTML templates
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]  # For images etc


# -- Options for MyST Parser -------------------------------------------------
# https://myst-parser.readthedocs.io/en/latest/index.html

myst_enable_extensions = [
    "fieldlist",  # Support for field lists (e.g. for function parameters)
    "dollarmath",  # Support for LaTeX-style math using $...$ and $$...$$
    "colon_fence",  # Support for callouts/admonitions (e.g. :::{important})
]


# -- Options for Napoleon -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

# napoleon_numpy_docstring = True


# -- Options for autosummary ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html

autosummary_generate = True
autosummary_imported_members = True


# -- Options for intersphinx ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}
