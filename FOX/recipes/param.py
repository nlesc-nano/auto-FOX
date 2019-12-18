"""
FOX.recipes.param
=================

A set of functions for analyzing and plotting ARMC results.

Examples
--------
A General overview of the functions within this module.

.. code:: python

    >>> import pandas as pd
    >>> from FOX.recipes import get_best, overlay_descriptor, plot_descriptor

    >>> hdf5_file: str = ...

    >>> param: pd.Series = get_best(hdf5_file, name='param')  # Extract the best parameters
    >>> rdf: pd.DataFrame = get_best(hdf5_file, name='rdf')  # Extract the matching RDF

    # Compare the RDF to its reference RDF and plot
    >>> rdf_dict = overlay_descriptor(hdf5_file, name='rdf')
    >>> plot_descriptor(rdf_dict)

.. image:: rdf.png
    :scale: 20 %
    :align: center


Examples
--------
An small workflow for calculating for calculating free energies using distribution functions
such as the radial distribution function (RDF).

.. code:: python

    >>> import pandas as pd
    >>> from FOX import get_free_energy
    >>> from FOX.recipes import get_best, overlay_descriptor, plot_descriptor

    >>> hdf5_file: str = ...

    >>> rdf: pd.DataFrame = get_best(hdf5_file, name='rdf')
    >>> G: pd.DataFrame = get_free_energy(rdf, unit='kcal/mol')

    >>> rdf_dict = overlay_descriptor(hdf5_file, name='rdf)
    >>> G_dict = {key: get_free_energy(value) for key, value in rdf_dict.items()}
    >>> plot_descriptor(G_dict)

.. image:: G_rdf.png
    :scale: 20 %
    :align: center


Examples
--------
An workflow for plotting parameters as a function of ARMC iterations.

.. code:: python

    >>> import pandas as pd
    >>> from FOX import from_hdf5
    >>> from FOX.recipes import plot_descriptor

    >>> hdf5_file: str = ...

    >>> param: pd.DataFrame = from_hdf5(hdf5_file, 'param')
    >>> param.index.name = 'ARMC iteration'
    >>> param_dict = {key: param[key] for key in param.columns.levels[0]}

    >>> plot_descriptor(param_dict)

.. image:: param.png
    :scale: 20 %
    :align: center


Index
-----
.. currentmodule:: FOX.recipes.param
.. autosummary::
    get_best
    overlay_descriptor
    plot_descriptor

API
---
.. autofunction:: get_best
.. autofunction:: overlay_descriptor
.. autofunction:: plot_descriptor

"""

from typing import Dict, Union, Iterable
from collections import abc

import pandas as pd

try:
    import matplotlib.pyplot as plt
    PltFigure = plt.Figure
    PLT_ERROR = None
except ImportError:
    PltFigure = 'matplotlib.pyplot.Figure'
    PLT_ERROR = ("Use of the FOX.{} function requires the 'matplotlib' package."
                 "\n'matplotlib' can be installed via PyPi with the following command:"
                 "\n\tpip install matplotlib")
try:
    import h5py
    H5PY_ERROR = None
except ImportError:
    H5PY_ERROR = ("Use of the FOX.{} function requires the 'h5py' package."
                  "\n'h5py' can be installed via conda with the following command:"
                  "\n\tconda install -n FOX -c conda-forge h5py")

from FOX import from_hdf5, assert_error

__all__ = ['get_best', 'overlay_descriptor', 'plot_descriptor']

NDFrame: type = pd.DataFrame.__bases__[0]  # Superclass of pd.DataFrame & pd.Series


@assert_error(H5PY_ERROR)
def get_best(hdf5_file: str, name: str = 'rdf', i: int = 0) -> pd.DataFrame:
    """Return the PES descriptor or ARMC property which yields the lowest error.

    Parameters
    ----------
    hdf5_file : :class:`str`
        The path+filename of the ARMC .hdf5 file.

    name : :class:`str`
        The name of the PES descriptor, *e.g.* ``"rdf"``.
        Alternatively one can supply an ARMC property such as ``"acceptance"``,
        ``"param"`` or ``"aux_error"``.

    i : :class:`int`
        The index of the desired PES.
        Only relevant for PES-descriptors of state-averaged ARMCs.

    Returns
    -------
    :class:`pandas.DataFrame` or :class:`pd.Series`
        A DataFrame of the optimal PES descriptor or other (user-specified) ARMC property.

    """
    full_name = f'{name}.{i}'
    with h5py.File(hdf5_file, 'r', libver='latest') as f:
        if full_name not in f.keys():  # i.e. if **name** does not belong to a PE descriptor
            full_name = name

    # Load the DataFrames
    if full_name == 'aux_error':
        aux_error = prop = from_hdf5(hdf5_file, 'aux_error')
    else:
        hdf5_dict = from_hdf5(hdf5_file, ['aux_error', full_name])
        aux_error, prop = hdf5_dict['aux_error'], hdf5_dict[full_name]

    # Return the best DataFrame (or Series)
    j: int = aux_error.sum(axis=1, skipna=False).idxmin()
    df = prop[j] if not isinstance(prop, NDFrame) else prop.iloc[j]
    if isinstance(df, pd.DataFrame):
        df.columns.name = full_name
    elif isinstance(df, pd.Series):
        df.name = full_name
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

    j: int = aux_error.sum(axis=1, skipna=False).idxmin()
    mm = mm[j]
    qm = qm[0]

    ret = {}
    for key in mm:
        df = pd.DataFrame({'MM-MD': mm[key], 'QM-MD': qm[key]}, index=mm.index)
        df.columns.name = mm_name
        ret[key] = df
    return ret


@assert_error(PLT_ERROR)
def plot_descriptor(descriptor: Union[NDFrame, Iterable[NDFrame]],
                    show_fig: bool = True) -> PltFigure:
    """Plot a DataFrame or iterable consisting of one or more DataFrames.

    Requires the ``matploblib`` package.

    Parameters
    ----------
    descriptor : :class:`pandas.DataFrame` or :class:`Iterable<collections.abc.Iterable>` [:class:`pandas.DataFrame`]
        A DataFrame or an iterable consisting of DataFrames.

    show_fig : :class:`bool`
        Whether to show the figure or not.

    Returns
    -------
    :class:`Figure<matplotlib.figure.Figure>`
        A matplotlib Figure.

    See Also
    --------
    :func:`get_best`
        Return the PES descriptor or ARMC property which yields the lowest error.

    :func:`overlay_descriptor`
        Return the PES descriptor which yields the lowest error and
        overlay it with the reference PES descriptor.

    """  # noqa
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
        if isinstance(key, tuple):
            key = ' '.join(repr(i) for i in key)
        df.plot(ax=ax, title=key, figsize=figsize)

    if show_fig:
        plt.show(block=True)
    return fig
