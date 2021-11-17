#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# qmflows documentation build configuration file, created by
# sphinx-quickstart on Wed Nov  8 12:07:40 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import datetime
sys.path.insert(0, os.path.abspath('..'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '2.4'


# This value controls how to represents typehints. The setting takes the following values:
# 'signature' – Show typehints as its signature (default)
# 'description' – Show typehints as content of function or method
# 'none' – Do not show typehints
autodoc_typehints = 'none'


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.duration',
    'matplotlib.sphinxext.plot_directive'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'Auto-FOX'
_year = str(datetime.datetime.now().year)
author = 'B. F. van Beek'
copyright = f'{_year}, {author}'


# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
release = '0.10.0'  # The full version, including alpha/beta/rc tags.
version = release.rsplit('.', maxsplit=1)[0]  # The short X.Y version.


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {'includehidden': False}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
        'donate.html',
    ]
}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'foxdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'FOX.tex', 'Auto-FOX Documentation', 'B. F. van Beek', 'manual')
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'Auto-FOX', 'Auto-FOX Documentation', [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'Auto-FOX', 'Auto-FOX Documentation', author, 'Auto-FOX',
     'Auto-FOX is a library for analyzing potential energy surfaces (PESs) and \
     using the resulting PES descriptors for constructing forcefield parameters.',
     'Miscellaneous')
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('http://matplotlib.org', None),
    'plams': ('https://www.scm.com/doc/plams/', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase/', None)
}


# Whether to show links to the files in HTML.
plot_html_show_formats = False


# Whether to show a link to the source in HTML.
plot_html_show_source_link = False


# File formats to generate. List of tuples or strings:
plot_formats = [('png', 300)]


# This value selects if automatically documented members are sorted alphabetical (value 'alphabetical'), by member type (value 'groupwise') or by source order (value 'bysource').
autodoc_member_order = 'bysource'


# This value controls the behavior of sphinx-build -W during importing modules.
autodoc_warningiserror = True


# This value contains a list of modules to be mocked up.
# This is useful when some external dependencies are not met at build time and break the building process.
# You may only specify the root package of the dependencies themselves and omit the sub-modules:
autodoc_mock_imports = ['h5py', 'rdkit', 'CAT']


# True to parse Google style docstrings.
# False to disable support for Google style docstrings.
# Defaults to True.
napoleon_google_docstring = True


# True to parse NumPy style docstrings.
# False to disable support for NumPy style docstrings.
# Defaults to True.
napoleon_numpy_docstring = True


# True to use the .. admonition:: directive for the Example and Examples sections.
# False to use the .. rubric:: directive instead. One may look better than the other depending on what HTML theme is used.
# Defaults to False.
napoleon_use_admonition_for_examples = True

# True to use the :ivar: role for instance variables.
# False to use the .. attribute:: directive instead.
#  Defaults to False.
napoleon_use_ivar = False

# A string of reStructuredText that will be included at the end of every source file that is read.
# This is a possible place to add substitutions that should be available in every file (another being rst_prolog).
rst_epilog = """
.. _FOX.MultiMolecule: 3_multimolecule.html#api
.. _FOX.MonteCarlo: 4_monte_carlo.html#fox-montecarlo-api
.. _FOX.ARMC: 4_monte_carlo.html#fox-armc-api
.. _FOX.PSF: https://github.com/nlesc-nano/auto-FOX/blob/master/FOX/classes/psf.py
.. _plams.PeriodicTable: https://www.scm.com/doc/plams/components/utils.html#periodic-table
.. _plams.Job: https://www.scm.com/doc/plams/components/jobs.html#job-api
.. _plams.Settings: https://www.scm.com/doc/plams/components/settings.html
.. _plams.Molecule: https://www.scm.com/doc/plams/components/mol_api.html
.. _np.ndarray: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
.. _np.float64: https://docs.scipy.org/doc/numpy/user/basics.types.html#array-types-and-conversions-between-types
.. _np.int64: https://docs.scipy.org/doc/numpy/user/basics.types.html#array-types-and-conversions-between-types
.. _pd.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
.. _pd.Series: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
.. _pd.Int64Index: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.Int64Index.html
.. _pd.MultiIndex: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.MultiIndex.html
.. _dict: https://docs.python.org/3/library/stdtypes.html#dict
.. _list: https://docs.python.org/3/library/stdtypes.html#list
.. _tuple: https://docs.python.org/3/library/stdtypes.html#tuple
.. _str: https://docs.python.org/3/library/stdtypes.html#str
.. _int: https://docs.python.org/3/library/functions.html#int
.. _float: https://docs.python.org/3/library/functions.html#float
.. _bool: https://docs.python.org/3/library/functions.html#bool
.. _type: https://docs.python.org/3/library/functions.html#type
.. _None: https://docs.python.org/3/library/constants.html#None
.. _Callable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable
.. _Sequence: https://docs.python.org/3/library/collections.abc.html#collections.abc.Sequence
.. _MutableSequence: https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableSequence
.. _Hashable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Hashable
.. _Iterable: https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable
.. _object: https://docs.python.org/3/library/functions.html#object

.. |FOX.ARMC| replace:: *FOX.ARMC*
.. |FOX.MonteCarlo| replace:: *FOX.MonteCarlo*
.. |FOX.MultiMolecule| replace:: *FOX.MultiMolecule*
.. |FOX.PSF| replace:: *FOX.PSF*
.. |plams.Job| replace:: *plams.Job*
.. |plams.Molecule| replace:: *plams.Molecule*
.. |plams.Settings| replace:: *plams.Settings*
.. |np.ndarray| replace:: *np.ndarray*
.. |np.float64| replace:: *np.float64*
.. |np.int64| replace:: *np.int64*
.. |pd.DataFrame| replace:: *pd.DataFrame*
.. |pd.Series| replace:: *pd.Series*
.. |pd.Int64Index| replace:: *pd.Int64Index*
.. |pd.MultiIndex| replace:: *pd.MultiIndex*
.. |dict| replace:: *dict*
.. |list| replace:: *list*
.. |tuple| replace:: *tuple*
.. |str| replace:: *str*
.. |int| replace:: *int*
.. |float| replace:: *float*
.. |bool| replace:: *bool*
.. |type| replace:: *type*
.. |None| replace:: *None*
.. |Callable| replace:: *Callable*
.. |Sequence| replace:: *Sequence*
.. |MutableSequence| replace:: *MutableSequence*
.. |Hashable| replace:: *Hashable*
.. |Iterable| replace:: *Iterable*
.. |object| replace:: *object*

"""

"""
.. |MutableMapping| replace:: :class:`MutableMapping<collections.abc.MutableMapping>`
.. |Mapping| replace:: :class:`Mapping<collections.abc.Mapping>`
.. |Iterable| replace:: :class:`Iterable<collections.abc.Iterable>`
.. |Iterator| replace:: :class:`Iterator<collections.abc.Iterator>`
.. |Collection| replace:: :class:`Collection<collections.abc.Collection>`
.. |Sequence| replace:: :class:`Sequence<collections.abc.Sequence>`
.. |Hashable| replace:: :class:`Hashable<collections.abc.Hashable>`
.. |Any| replace :data:`Any<typing.Any>`
"""
