# This code is part of Qiskit.
#
# (C) Copyright IBM 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Sphinx documentation builder."""

# -- General configuration ---------------------------------------------------
import datetime
import doctest

project = "Qiskit"
copyright = f"2017-{datetime.date.today().year}, Qiskit Development Team"  # pylint: disable=redefined-builtin
author = "Qiskit Development Team"

# The short X.Y version
version = "0.25"
# The full version, including alpha/beta/rc tags
release = "0.25.0"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "reno.sphinxext",
    "sphinx_design",
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.doctest"
]

templates_path = ["_templates"]

# Number figures, tables and code-blocks if they have a caption.
numfig = True
# Available keys are 'figure', 'table', 'code-block' and 'section'.  '%s' is the number.
numfig_format = {"table": "Table %s"}

# The language for content autogenerated by Sphinx or the default for gettext content translation.
language = "en"

# Relative to source directory, affects general discovery, and html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

pygments_style = "colorful"

# Whether module names are included in crossrefs of functions, classes, etc.
add_module_names = False

# A list of prefixes that are ignored for sorting the Python module index
# (e.g., if this is set to ['foo.'], then foo.bar is shown under B, not F).
modindex_common_prefix = ["qiskit."]

intersphinx_mapping = {
    "retworkx": ("https://qiskit.org/documentation/retworkx/", None),
    "qiskit-ibm-runtime": ("https://qiskit.org/documentation/partners/qiskit_ibm_runtime/", None),
    "qiskit-aer": ("https://qiskit.org/documentation/aer/", None),
    "numpy": ("https://numpy.org/doc/stable/", None)
}

# -- Options for HTML output -------------------------------------------------

html_theme = "qiskit_sphinx_theme"
html_last_updated_fmt = "%Y/%m/%d"
html_theme_options = {
    "logo_only": True,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
}


# -- Options for Autosummary and Autodoc -------------------------------------

# Note that setting autodoc defaults here may not have as much of an effect as you may expect; any
# documentation created by autosummary uses a template file (in autosummary in the templates path),
# which likely overrides the autodoc defaults.

# Move type hints from signatures to the parameter descriptions (except in overload cases, where
# that's not possible).
autodoc_typehints = "description"
# Only add type hints from signature to description body if the parameter has documentation.  The
# return type is always added to the description (if in the signature).
autodoc_typehints_description_target = "documented_params"

autosummary_generate = True
autosummary_generate_overwrite = False

# The pulse library contains some names that differ only in capitalisation, during the changeover
# surrounding SymbolPulse.  Since these resolve to autosummary filenames that also differ only in
# capitalisation, this causes problems when the documentation is built on an OS/filesystem that is
# enforcing case-insensitive semantics.  This setting defines some custom names to prevent the clash
# from happening.
autosummary_filename_map = {
    "qiskit.pulse.library.Constant": "qiskit.pulse.library.Constant_class.rst",
    "qiskit.pulse.library.Sawtooth": "qiskit.pulse.library.Sawtooth_class.rst",
    "qiskit.pulse.library.Triangle": "qiskit.pulse.library.Triangle_class.rst",
    "qiskit.pulse.library.Cos": "qiskit.pulse.library.Cos_class.rst",
    "qiskit.pulse.library.Sin": "qiskit.pulse.library.Sin_class.rst",
    "qiskit.pulse.library.Gaussian": "qiskit.pulse.library.Gaussian_class.rst",
    "qiskit.pulse.library.Drag": "qiskit.pulse.library.Drag_class.rst",
}

autoclass_content = "both"

# -- Options for Doctest --------------------------------------------------------

import sphinx.ext.doctest

# This option will make doctest ignore whitespace when testing code.
# It's specially important for circuit representation as it gives an
# error otherwise
doctest_default_flags = sphinx.ext.doctest.doctest.NORMALIZE_WHITESPACE

# Leaving this string empty disables testing of doctest blocks from docstrings.
# Doctest blocks are structures like this one:
# >> code
# output
doctest_test_doctest_blocks = ""
