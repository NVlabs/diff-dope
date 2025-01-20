# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os

# Ensure Sphinx uses the correct Python interpreter
# python_interpreter = os.path.join(sys.base_exec_prefix, 'bin', 'python')
# if os.path.exists(python_interpreter):
#     sys.executable = python_interpreter
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "Diff-DOPE"
copyright = "2023, Jonathan Tremblay, et al."
author = "Jonathan Tremblay, et al."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]


# In your conf.py file
def autodoc_skip_member(app, what, name, obj, skip, options):
    # Ref: https://stackoverflow.com/a/21449475/
    exclusions = (
        "make_grid",  # special-members
        "__doc__",
        "__module__",
        "__dict__",  # undoc-members
    )
    exclude = name in exclusions
    # return True if (skip or exclude) else None  # Can interfere with subsequent skip functions.
    return True if exclude else None


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)


import sys

print("Python Version:")
print(sys.version)
