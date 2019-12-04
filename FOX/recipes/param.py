"""
FOX.recipes.param
=================

A set of functions for analyzing and plotting ARMC results.

Examples
--------
.. code:: python

    >>> import pandas as pd
    >>> from FOX.recipes import (
    ...     get_best_param, get_best_descriptor, overlay_descriptor, plot_descriptor
    ... )

    >>> hdf5_file: str = ...

    >>> param: pd.Series = get_best_param(hdf5_file)  # Extract the best parameters
    >>> rdf: pd.DataFrame = get_best_descriptor(hdf5_file, name='rdf')  # Extract the matching RDF

    # Compare the RDF to its reference RDF and plot
    >>> rdf_dict = overlay_descriptor(hdf5_file, name='rdf')
    >>> plot_descriptor(rdf_dict)

.. image:: rdf.png
    :scale: 20 %
    :align: center

Index
-----
.. currentmodule:: FOX.recipes.param
.. autosummary::
    get_best_param
    get_best_descriptor
    overlay_descriptor
    plot_descriptor

API
---
.. autofunction:: get_best_param
.. autofunction:: get_best_descriptor
.. autofunction:: overlay_descriptor
.. autofunction:: plot_descriptor

"""

from typing import Dict, Union, Iterable
from collections import abc

import pandas as pd

try:
    import matplotlib.pyplot as plt
    PltFigure = plt.Figure
    PLT_ERROR = ''
except ImportError:
    PltFigure = 'matplotlib.pyplot.Figure'
    PLT_ERROR = ("Use of the FOX.{} function requires the 'matplotlib' package."
                 "\n'matplotlib' can be installed via PyPi with the following command:"
                 "\n\tpip install matplotlib")

from FOX import from_hdf5, assert_error

__all__ = ['get_best_param', 'get_best_descriptor', 'overlay_descriptor', 'plot_descriptor']


def get_best_param(hdf5_file: str) -> pd.Series:
    """Return the parameter set which yields the lowest error.

    Parameters
    ----------
    hdf5_file : :class:`str`
        The path+filename of the ARMC .hdf5 file.

    Returns
    -------
    :class:`pandas.Series`
        A Series with the optimal ARMC parameters.

    """
    hdf5_dict = from_hdf5(hdf5_file, ['aux_error', 'param'])
    aux_error, param = hdf5_dict['aux_error'], hdf5_dict['param']

    i: int = aux_error.sum(axis=1).idxmin()
    return param.loc[i]


def get_best_descriptor(hdf5_file: str, name: str = 'rdf', i: int = 0) -> pd.DataFrame:
    """Return the PES descriptor which yields the lowest error.

    Parameters
    ----------
    hdf5_file : :class:`str`
        The path+filename of the ARMC .hdf5 file.

    name : :class:`str`
        The name of the PES descriptor, *e.g.* ``"rdf"``.

    i : :class:`int`
        The index of desired PES.
        Only relevant for state-averaged ARMCs.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame of the optimal PES descriptor.

    """
    full_name = f'{name}.{i}'
    hdf5_dict = from_hdf5(hdf5_file, ['aux_error', full_name])
    aux_error, descr = hdf5_dict['aux_error'], hdf5_dict[full_name]

    j: int = aux_error.sum(axis=1).idxmin()
    df = descr[j]
    df.columns.name = full_name
    return df


def overlay_descriptor(hdf5_file: str, name: str = 'rdf', i: int = 0) -> Dict[str, pd.DataFrame]:
    """Return the PES descriptor which yields the lowest error and overlay it with the reference PES descriptor.

    Parameters
    ----------
    hdf5_file : :class:`str`
        The path+filename of the ARMC .hdf5 file.

    name : :class:`str`
        The name of the PES descriptor, *e.g.* ``"rdf"``.

    i : :class:`int`
        The index of desired PES.
        Only relevant for state-averaged ARMCs.

    Returns
    -------
    :class:`dict` [:class:`str`, :class:`pandas.DataFrame`]
        A dictionary of DataFrames.
        Values consist of DataFrames with two keys: ``"MM-MD"`` and ``"QM-MD"``.
        Atom pairs, such as ``"Cd Cd"``, are used as keys.

    """  # noqa
    mm_name = f'{name}.{i}'
    qm_name = f'{name}.{i}.ref'
    hdf5_dict = from_hdf5(hdf5_file, ['aux_error', mm_name, qm_name])
    aux_error, mm, qm = hdf5_dict['aux_error'], hdf5_dict[mm_name], hdf5_dict[qm_name]

    j: int = aux_error.sum(axis=1).idxmin()
    mm = mm[j]
    qm = qm[0]

    ret = {}
    for key in mm:
        df = pd.DataFrame({'MM-MD': mm[key], 'QM-MD': qm[key]}, index=mm.index)
        df.columns.name = mm_name
        ret[key] = df
    return ret


DF = Union[pd.Series, pd.DataFrame, Iterable[pd.DataFrame], Iterable[pd.Series]]


@assert_error(PLT_ERROR)
def plot_descriptor(descriptor: DF) -> PltFigure:
    """Plot a DataFrame or iterable consisting of one or more DataFrames.

    Requires the ``matploblib`` package.

    Parameters
    ----------
    descriptor :class:`pandas.DataFrame`, :class:`pandas.Series` or
    :class:`Iterable<collections.abc.Iterable>` [:class:`pandas.DataFrame`]
        A DataFrame or an iterable consisting of DataFrames.

    Returns
    -------
    :class:`Figure<matplotlib.pyplot.Figure>`
        A matplotlib figure.

    See Also
    --------
    :func:`get_best_descriptor`
        Return the PES descriptor which yields the lowest error.

    :func:`overlay_descriptor`
        Overlay the PES descriptor, which yields the lowest error, with its QM reference.

    """
    if isinstance(descriptor, pd.Series):
        descriptor = descriptor.to_frame()

    if isinstance(descriptor, pd.DataFrame):
        ncols = len(descriptor.columns)
        iterator = descriptor.items()
    elif isinstance(descriptor, abc.Mapping):
        ncols = len(descriptor)
        iterator = descriptor.items()
    else:
        try:
            ncols = len(descriptor)
        except TypeError:  # It's an iterator
            descriptor = list(descriptor)
            ncols = len(descriptor)
        iterator = enumerate(descriptor)

    figsize = (4 * ncols, 6)

    fig, ax_tup = plt.subplots(ncols=ncols, sharex=True, sharey=False)
    for (key, df), ax in zip(iterator, ax_tup):
        df.plot(ax=ax, title=key, figsize=figsize)

    plt.show(block=True)
    return fig
