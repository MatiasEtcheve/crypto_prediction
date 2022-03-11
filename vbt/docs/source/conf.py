# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config
# flake8: noqa

# -- Path setup --------------------------------------------------------------

import os
import sys
from datetime import date

import m2r

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import recommonmark
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath("../.."))

import numpy as np

np.set_printoptions(precision=2, threshold=5)


# -- Project information -----------------------------------------------------

project = "Binance"
copyright = f"2021, IronKitten"
author = "IronKitten"

# The full version, including alpha/beta/rc tags
release = "0.0.1™"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.linkcode",
    "recommonmark",
    "autodocsumm",
    "sphinx_autodoc_typehints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = 'sphinxdoc'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# We keep the line below as a comment as will be default config value when we'll add static files

# html_css_files = ["css/custom.css"]

html_theme_options = {
    "canonical_url": "",
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    # "style_nav_header_background": "#049241",
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}


# -- Extension configuration -------------------------------------------------


def docstrings(app, what, name, obj, options, lines):
    text = m2r.convert("\n".join(lines))
    lines.clear()
    for line in text.splitlines():
        lines.append(line)


def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            #'url_resolver': lambda url: url,
            "enable_auto_toc_tree": False,
            "auto_toc_tree_section": "Contents",
        },
        True,
    )
    app.add_transform(AutoStructify)
    app.connect("autodoc-process-docstring", docstrings)



source_suffix = [".rst", ".md"]

add_module_names = False


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return "https://github.com/scortexio/sensei/tree/master/%s.py" % filename


autodoc_default_options = {"autosummary": True}


# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
autoclass_content = "both"
