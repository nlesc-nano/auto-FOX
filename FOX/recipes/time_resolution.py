"""A set of functions for calculating time-resolved distribution functions.

Index
-----
.. currentmodule:: FOX.recipes
.. autosummary::
    time_resolved_rdf
    time_resolved_adf

API
---
.. autofunction:: time_resolved_rdf
.. autofunction:: time_resolved_adf

"""


from typing import List, Optional, Any, Callable

import FOX
import numpy as np
import pandas as pd

__all__ = ["time_resolved_rdf", "time_resolved_adf"]


def _time_resolved(
    mol: FOX.MultiMolecule,
    func: Callable[..., pd.DataFrame],
    start: int,
    stop: int,
    step: int,
    **kwargs: Any,
) -> List[pd.DataFrame]:
    """Calculate a time-resolved property."""
    i0 = start
    i1 = step
    ret = []

    while i0 < stop:
        df = func(mol, mol_subset=np.s_[i0:i1], **kwargs)
        ret.append(df)
        i0 += step
        i1 += step

    return ret


def time_resolved_rdf(mol: FOX.MultiMolecule, start: int = 0, stop: Optional[int] = None,
                      step: int = 500, **kwargs: Any) -> List[pd.DataFrame]:
    r"""Calculate the time-resolved radial distribution function (RDF).

    Examples
    --------
    .. code-block:: python

        >>> from FOX import MultiMolecule, example_xyz
        >>> from FOX.recipes import time_resolved_rdf

        # Calculate each RDF over the course of 500 frames
        >>> time_step = 500
        >>> mol = MultiMolecule.from_xyz(example_xyz)

        >>> rdf_list = time_resolved_rdf(
        ...     mol, step=time_step, atom_subset=['Cd', 'Se']
        ... )

    Parameters
    ----------
    mol : :class:`~FOX.MultiMolecule`
        The trajectory in question.
    start : :class:`int`
        The initial frame.
    stop : :class:`int`, optional
        The final frame.
        Set to :data:`None` to iterate over all frames.
    step : :class:`int`
        The number of frames per individual RDF.
        Note that lower **step** values will result in increased numerical noise.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :meth:`~FOX.MultiMolecule.init_rdf`.

    Returns
    -------
    :class:`List[pandas.DataFrame]<typing.List>`
        A list of dataframes, each containing an RDF calculated over the course
        of **step** frames.

    See Also
    --------
    :meth:`~FOX.MultiMolecule.init_rdf`
        Calculate the radial distribution function.

    """
    func = FOX.MultiMolecule.init_rdf
    stop_ = stop if stop is not None else len(mol)
    return _time_resolved(mol, func, start, stop_, step, **kwargs)


def time_resolved_adf(mol: FOX.MultiMolecule, start: int = 0, stop: Optional[int] = None,
                      step: int = 500, **kwargs: Any) -> List[pd.DataFrame]:
    r"""Calculate the time-resolved angular distribution function (ADF).

    Examples
    --------
    .. code-block:: python

        >>> from FOX import MultiMolecule, example_xyz
        >>> from FOX.recipes import time_resolved_adf

        # Calculate each ADF over the course of 500 frames
        >>> time_step = 500
        >>> mol = MultiMolecule.from_xyz(example_xyz)

        >>> rdf_list = time_resolved_adf(
        ...     mol, step=time_step, atom_subset=['Cd', 'Se']
        ... )

    Parameters
    ----------
    mol : :class:`~FOX.MultiMolecule`
        The trajectory in question.
    start : :class:`int`
        The initial frame.
    stop : :class:`int`, optional
        The final frame.
        Set to :data:`None` to iterate over all frames.
    step : :class:`int`
        The number of frames per individual RDF.
        Note that lower **step** values will result in increased numerical noise.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :meth:`~FOX.MultiMolecule.init_adf`.

    Returns
    -------
    :class:`List[pandas.DataFrame]<typing.List>`
        A list of dataframes, each containing an ADF calculated over the course
        of **step** frames.

    See Also
    --------
    :meth:`~FOX.MultiMolecule.init_adf`
        Calculate the angular distribution function.

    """
    func = FOX.MultiMolecule.init_adf
    stop_ = stop if stop is not None else len(mol)
    return _time_resolved(mol, func, start, stop_, step, **kwargs)
