"""A set of functions for analyzing and plotting ARMC results.

Examples
--------
A general overview of the functions within this module.

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
A small workflow for calculating for calculating free energies using distribution functions
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
A workflow for plotting parameters as a function of ARMC iterations.

.. code:: python

    >>> import numpy as np
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

This approach can also be used for the plotting of other properties such as the auxiliary error.

.. code:: python

    >>> ...

    >>> err: pd.DataFrame = from_hdf5(hdf5_file, 'aux_error')
    >>> err.index.name = 'ARMC iteration'
    >>> err_dict = {'Auxiliary Error': err}

    >>> plot_descriptor(err_dict)

.. image:: err.png
    :scale: 20 %
    :align: center

On occasion it might be desirable to only print the error of, for example, accepted iterations.
Given a sequence of booleans (``bool_seq``), one can slice a DataFrame or Series (``df``) using
:code:`df.loc[bool_seq]`.

.. code:: python

    >>> ...

    >>> acceptance: np.ndarray = from_hdf5(hdf5_file, 'acceptance')  # Boolean array
    >>> err_slice_dict = {key: df.loc[acceptance], value for key, df in err_dict.items()}

    >>> plot_descriptor(err_slice_dict)


Index
-----
.. currentmodule:: FOX.recipes
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

from os import PathLike
from typing import Dict, Union, Iterable, Any, FrozenSet, Tuple, Iterator, cast, Optional, Mapping
from collections import abc

import h5py
import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from nanoutils import raise_if

from FOX import from_hdf5
from FOX.logger import DEFAULT_LOGGER as logger

try:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import Figure
    PLT_ERROR: Optional[ImportError] = None
except ImportError as ex:
    from FOX.type_alias import Figure
    PLT_ERROR = ex

__all__ = ['get_best', 'overlay_descriptor', 'plot_descriptor']

PlotAccessor: type = pd.DataFrame.plot  # A class used by Pandas for plotting stuff


def get_best(hdf5_file: Union[str, 'PathLike[str]'],
             name: str = 'rdf', i: int = 0) -> pd.DataFrame:
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
        shape = f['aux_error'].shape[:2]
    if full_name.startswith('/'):
        full_name = full_name.strip('/')

    # Load the DataFrames
    if full_name in '/aux_error':
        aux_error = prop = from_hdf5(hdf5_file, 'aux_error')
    else:
        hdf5_dict = from_hdf5(hdf5_file, ['aux_error', full_name])
        aux_error, prop = hdf5_dict['aux_error'], hdf5_dict[full_name]

    # Return the best DataFrame (or Series)
    j: int = aux_error.sum(axis=1, skipna=False).idxmin()
    logger.debug(f"Optimum ARMC cycle: {np.unravel_index(j, shape)}")

    df = prop[j] if not isinstance(prop, NDFrame) else prop.iloc[j]
    if isinstance(df, pd.DataFrame):
        df.columns.name = full_name
    elif isinstance(df, pd.Series):
        df.name = full_name
    return df


def overlay_descriptor(hdf5_file: Union[str, 'PathLike[str]'], name: str = 'rdf',
                       i: int = 0) -> Dict[str, pd.DataFrame]:
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
    with h5py.File(hdf5_file, 'r', libver='latest') as f:
        shape = f['aux_error'].shape[:2]

    mm_name = f'{name}.{i}'
    qm_name = f'{name}.{i}.ref'
    hdf5_dict = from_hdf5(hdf5_file, ['aux_error', mm_name, qm_name])
    aux_error, mm, qm = hdf5_dict['aux_error'], hdf5_dict[mm_name], hdf5_dict[qm_name]

    j: int = aux_error.sum(axis=1, skipna=False).idxmin()
    logger.debug(f"Optimum ARMC cycle: {np.unravel_index(j, shape)}")
    mm = mm[j]
    qm = qm[0]

    ret = {}
    for key in mm:
        df = pd.DataFrame({'MM-MD': mm[key], 'QM-MD': qm[key]}, index=mm.index)
        df.columns.name = mm_name
        ret[key] = df
    return ret


@raise_if(PLT_ERROR)
def plot_descriptor(descriptor: Union[NDFrame, Iterable[NDFrame]],
                    show_fig: bool = True, kind: str = 'line',
                    sharex: bool = True, sharey: bool = False,
                    **kwargs: Any) -> Figure:
    r"""Plot a DataFrame or iterable consisting of one or more DataFrames.

    Requires the matplotlib_ package.

    .. _matplotlib: https://matplotlib.org/

    Parameters
    ----------
    descriptor : :class:`pandas.DataFrame` or :class:`Iterable<collections.abc.Iterable>` [:class:`pandas.DataFrame`]
        A DataFrame or an iterable consisting of DataFrames.

    show_fig : :class:`bool`
        Whether to show the figure or not.

    kind : :class:`str`
        The plot kind to-be passed to :meth:`pandas.DataFrame.plot`.

    sharex/sharey : :class:`bool`
        Whether or not the to-be created plots should share their x/y-axes.

    \**kwargs : :data:`Any<typing.Any>`
        Further keyword arguments for the :meth:`pandas.DataFrame.plot` method.

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
    kind_ = _validate_kind(kind)
    ncols, iterator = _get_df_iterator(descriptor)

    figsize = (4 * ncols, 6)
    fig, ax_tup = plt.subplots(ncols=ncols, sharex=sharex, sharey=sharey)
    if ncols == 1:  # Ensure ax_tup is actually a tuple
        ax_tup = (ax_tup,)

    # Construct the actual plots
    for (key, df), ax in zip(iterator, ax_tup):
        if isinstance(key, tuple):
            key = ' '.join(repr(i) for i in key)
        df.plot(ax=ax, title=key, figsize=figsize, kind=kind_, **kwargs)

    if show_fig:
        plt.show(block=True)
    return fig


#: A :class:`frozenset` with valid values for the **kind** parameter in :func:`plot_descriptor`.
VALID_KIND: FrozenSet[str] = frozenset(
    PlotAccessor._all_kinds + tuple(PlotAccessor._kind_aliases.values())  # type: ignore[attr-defined] # noqa: E501
)


def _validate_kind(kind: str) -> str:
    """Validate the **kind** parameter for :func:`plot_descriptor`."""
    try:
        ret = kind.lower()
    except AttributeError as ex:
        raise TypeError("'kind' expected a 'str'; observed type: "
                        f"'{kind.__class__.__name__}'").with_traceback(ex.__traceback__)
    if ret not in VALID_KIND:
        raise ValueError(f"{repr(ret)} is not a valid value for 'kind'; "
                         f"accepted values: {tuple(sorted(VALID_KIND))}")
    return ret


def _get_df_iterator(descriptor: Union[NDFrame, Mapping[Any, NDFrame], Iterable[NDFrame]]
                     ) -> Tuple[int, Iterator[Tuple[Any, NDFrame]]]:
    """Return the number of plots and a DataFrame enumerator for :func:`plot_descriptor`."""
    if isinstance(descriptor, pd.Series):
        descriptor = descriptor.to_frame()

    # Figure out the number of plots
    try:
        ncols: int = len(descriptor.keys())  # type: ignore[union-attr]
    except AttributeError:
        try:
            ncols = len(descriptor)  # type: ignore
        except TypeError as ex:
            if not isinstance(descriptor, abc.Iterable):
                tb = ex.__traceback__
                raise TypeError("'descriptor' expected an iterable; observed type: "
                                f"'{descriptor.__class__.__name__}'").with_traceback(tb)
            descriptor = list(descriptor)
            ncols = len(descriptor)

    # Construct an iterator of 2-tuples
    try:
        iterator = cast(Iterator[Tuple[Any, NDFrame]], descriptor.items())  # type: ignore[union-attr] # noqa: E501
    except (AttributeError, TypeError):
        iterator = enumerate(descriptor)

    return ncols, iterator
