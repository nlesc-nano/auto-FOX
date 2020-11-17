"""Functions related to the Lennard-Jones parameters of the Universal force field (UFF).

Index
-----
.. currentmodule:: FOX.ff.lj_uff
.. autosummary::
    UFF_DF
    combine_sigma
    combine_epsilon

API
---
.. autodata:: UFF_DF
    :annotation: = <pandas.core.frame.DataFrame object>

.. autofunction:: combine_sigma
.. autofunction:: combine_epsilon

"""

from os.path import join, dirname

import pandas as pd

__all__ = ['UFF_DF', 'combine_sigma', 'combine_epsilon']

#: Absolute path to the ``FOX.data.uff`` .csv file.
_CSV: str = join(dirname(dirname(__file__)), 'data', 'uff.csv')

#: A DataFrame with UFF Lennard-Jones parameters.
#: Has access to the ``"sigma"`` and ``"epsilon"``` columns.
#: See :data:`_CSV` for the path to the corresponding .csv file.
UFF_DF: pd.DataFrame = pd.read_csv(_CSV, index_col=0, skiprows=10)[['epsilon', 'sigma']]
UFF_DF['sigma'] /= 2**(1/6)
UFF_DF.columns.name = 'Å & kcal/mol'


def combine_sigma(at1: str, at2: str) -> float:
    r"""Return the arithmetic mean of two UFF Lennard-Jones distances (:math:`\sigma_{i}`).

    .. math::

        x_{ab} = \frac {\sigma_{a} + \sigma_{b}}{2}

    Distances are pulled from the ``"sigma"`` column in :data:`UFF_DF` based on the supplied
    atomic symbols (**a** and **b**).

    Paramaters
    ----------
    at1 : str
        The first atomic symbol.

    at2 : str
        The second atomic symbol.

    Raises
    ------
    ValueError:
        Raised if **a** and/or **b** cannot be found in the index of :data:`UFF_DF`.

    """
    try:
        sigma1 = UFF_DF.at[at1, 'sigma']
        sigma2 = UFF_DF.at[at2, 'sigma']
    except KeyError as ex:
        raise ValueError(f"No UFF parameters available for atom type {ex}") from None
    return (sigma1 + sigma2) / 2


def combine_epsilon(at1: str, at2: str) -> float:
    r"""Return the geometric mean of two UFF Lennard-Jones well depths (:math:`\varepsilon_{i}`).

    .. math::

        \varepsilon_{ab} = \sqrt{\varepsilon_{a} \varepsilon_{b}}

    Well depts are pulled from the ``"epsilon"`` column in :data:`UFF_DF` based on the supplied
    atomic symbols (**a** and **b**).

    Paramaters
    ----------
    at1 : str
        The first atomic symbol.

    at2 : str
        The second atomic symbol.

    Raises
    ------
    ValueError:
        Raised if **a** and/or **b** cannot be found in the index of :data:`UFF_DF`.

    """
    try:
        epsilon1 = UFF_DF.at[at1, 'epsilon']
        epsilon2 = UFF_DF.at[at2, 'epsilon']
    except KeyError as ex:
        raise ValueError(f"No UFF parameters available for atom type {ex}") from None
    return (epsilon1 * epsilon2)**0.5
