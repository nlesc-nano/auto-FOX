"""
FOX.armc.plotting
=================

A module with functions for analyzing ARMC results.

Index
-----
.. currentmodule:: FOX.armc.plotting
.. autosummary::
    plot_pes_descriptors
    plot_param
    plot_dset

API
---
.. autofunction:: plot_pes_descriptors
.. autofunction:: plot_param
.. autofunction:: plot_dset

"""

from typing import Optional, Iterable, Union, Hashable

import pandas as pd

from ..io.hdf5_utils import from_hdf5
from ..functions.utils import assert_error

try:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import Figure
    PLT_ERROR = None
except ImportError:
    from ..type_alias import Figure
    PLT_ERROR = ("Use of the FOX.{} function requires the 'matplotlib' package."
                 "\n'matplotlib' can be installed via PyPi with the following command:"
                 "\n\tpip install matplotlib")

__all__ = []


@assert_error(PLT_ERROR)
def plot_pes_descriptors(filename_in: str,
                         descriptor: str,
                         filename_out: Optional[str],
                         iteration: int = -1,
                         savefig_kwarg: Optional[dict] = None) -> Figure:
    """Create and save a figure showing side-by-side comparisons of QM and MM PES descriptors.

    .. _matplotlib.savefig: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.savefig.html

    Examples
    --------
    .. code:: python

        >>> import matplotlib.pyplot as plt

        >>> descriptor = 'rdf'
        >>> fig = plot_pes_descriptors('armc.hdf5', 'rdf.png', descriptor)
        >>> plt.show()

    .. image:: rdf.png
        :align: center

    Parameters
    ----------
    filename_in : str
        The path+name of the ARMC .hdf5 file.

    descriptor : str
        The name of the dataset containing the to-be retrieved PES descriptor (*e.g.* ``"RDF"``).

    filename_out : str
        Optional: The path+name of the to-be created figure.
        Will default to to all dataset names in **datasets** (appended with ``'.png'``) if ``None``.

    iteration : int
        The ARMC iteration containg the PES descriptor of interest.

    savefig_kwarg : |None|_ or |dict|_
        Optional: A dictionary with user-specified keyword arguments for matplotlib.savefig_.

    Returns
    -------
    |matplotlib.figure.Figure|_:
        A matplotlib figure containg PES descriptors.

    """
    filename_out = filename_out or descriptor + '.png'

    # Gather the pes descriptors
    pes_mm = from_hdf5(filename_in, descriptor)[iteration]
    pes_qm = from_hdf5(filename_in, descriptor + '.ref')[0]

    # Define constants
    ncols = len(pes_mm.columns)
    figsize = (4 * ncols, 6)

    # Construct the figures
    fig, ax_tup = plt.subplots(ncols=ncols, sharex=True, sharey=False)
    for key, ax in zip(pes_mm, ax_tup):
        df = pd.DataFrame({'MM-MD': pes_mm[key], 'QM-MD': pes_qm[key]}, index=pes_mm.index)
        df.columns.name = descriptor
        df.plot(ax=ax, title=key, figsize=figsize)

    # Save and return the figures
    if savefig_kwarg is None:
        plt.savefig(filename_out, dpi=300, quality=100, transparent=True, format='png')
    else:
        plt.savefig(filename_out, **savefig_kwarg)
    return fig


@assert_error(PLT_ERROR)
def plot_param(filename_in: str,
               filename_out: Optional[str] = None,
               savefig_kwarg: Optional[dict] = None) -> Figure:
    """Create and save a figure from the ``"param"`` hdf5 dataset in **filename_in**.

    .. _matplotlib.savefig: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.savefig.html

    Examples
    --------
    .. code:: python

        >>> import matplotlib.pyplot as plt

        >>> fig = plot_pes_descriptors('armc.hdf5', 'param.png')
        >>> plt.show()

    .. image:: param.png
        :align: center

    Parameters
    ----------
    filename_in : str
        The path+name of the ARMC .hdf5 file.

    filename_out : str
        Optional: The path+name of the to-be created figure.
        Will default to to ``'param.png'`` if ``None``.

    savefig_kwarg : |None|_ or |dict|_
        Optional: A dictionary with user-specified keyword arguments for matplotlib.savefig_.

    Returns
    -------
    |matplotlib.figure.Figure|_:
        A matplotlib figure containg forcefield parameters.

    """
    filename_out = filename_out or 'param.png'

    # Gather the pes descriptors
    param = from_hdf5(filename_in, 'param')

    # Define constants
    ncols = len(param.columns.levels[0])
    figsize = (4 * ncols, 6)

    # Construct the figures
    fig, ax_tup = plt.subplots(ncols=ncols, sharex=True, sharey=False)
    for key, ax in zip(param.columns.levels[0], ax_tup):
        df = param[key].copy()
        df.columns.name = 'Atoms/Atom pairs'
        df.index.name = 'ARMC Iteration'
        df.plot(ax=ax, title=key, figsize=figsize)

    # Save and return the figures
    if savefig_kwarg is None:
        plt.savefig(filename_out, dpi=300, quality=100, transparent=True, format='png')
    else:
        plt.savefig(filename_out, **savefig_kwarg)
    return fig


@assert_error(PLT_ERROR)
def plot_dset(filename_in: str,
              datasets: Union[Hashable, Iterable[Hashable]],
              filename_out: Optional[str] = None,
              savefig_kwarg: Optional[dict] = None) -> Figure:
    """Create and save a figure from an arbitrary hdf5 dataset in **filename_in**.

    See :func:`plot_pes_descriptors` and :func:`plot_param` for functions specialized in plotting
    forcefield parameters and PES descriptors, respectively.

    .. _matplotlib.savefig: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.savefig.html

    Examples
    --------
    .. code:: python

        >>> import matplotlib.pyplot as plt

        >>> datasets = ('aux_error', 'acceptance')
        >>> fig = plot_pes_descriptors('armc.hdf5', 'dsets.png', datasets)
        >>> plt.show()

    .. image:: datasets.png
        :align: center

    Parameters
    ----------
    filename_in : str
        The path+name of the ARMC .hdf5 file.

    dataset : |str|_ or |list|_[|str|_]
        The name of the dataset containing the to-be retrieved datasets (*e.g.* ``"aux_error"``).

    filename_out : str
        Optional: The path+name of the to-be created figure.
        Will default to ``'datasets.png'`` if ``None``.

    savefig_kwarg : |None|_ or |dict|_
        Optional: A dictionary with user-specified keyword arguments for matplotlib.savefig_.

    Returns
    -------
    |matplotlib.figure.Figure|_:
        A matplotlib figure containg PES descriptors.

    """
    filename_out = filename_out or 'datasets.png'

    if isinstance(datasets, str):
        datasets = [datasets]

    # Gather the pes descriptors
    df_dict = {dset: from_hdf5(filename_in, dset) for dset in datasets}
    for k, v in df_dict.items():
        if isinstance(v, list):
            df_dict[k] = v[-1]
        elif k == 'acceptance':
            df_dict[k] = v.astype(int)

    # Define constants
    ncols = len(df_dict)
    figsize = (4 * ncols, 6)

    # Construct the figures
    fig, ax_tup = plt.subplots(ncols=ncols, sharex=False, sharey=False)
    if ncols == 1:
        ax_tup = (ax_tup,)
    for (key, df), ax in zip(df_dict.items(), ax_tup):
        df.index.name = 'ARMC Iteration'
        df.plot(ax=ax, title=key, figsize=figsize)

    # Save and return the figures
    if savefig_kwarg is None:
        plt.savefig(filename_out, dpi=300, quality=100, transparent=True, format='png')
    else:
        plt.savefig(filename_out, **savefig_kwarg)
    return fig
