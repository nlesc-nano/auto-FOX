"""A module for estimating Lennard-Jones parameters.

Examples
--------
.. code:: python

    >>> import pandas as pd
    >>> from FOX import MultiMolecule, example_xyz, estimate_lennard_jones

    >>> xyz_file: str = example_xyz
    >>> atom_subset = ['Cd', 'Se', 'O']

    >>> mol = MultiMolecule.from_xyz(xyz_file)
    >>> rdf: pd.DataFrame = mol.init_rdf(atom_subset=atom_subset)
    >>> param: pd.DataFrame = estimate_lennard_jones(rdf)

    >>> print(param)
                sigma (Angstrom)  epsilon (kj/mol)
    Atom pairs
    Cd Cd                   3.95          2.097554
    Cd Se                   2.50          4.759017
    Cd O                    2.20          3.360966
    Se Se                   4.20          2.976106
    Se O                    3.65          0.992538
    O O                     2.15          6.676584


Index
-----
.. currentmodule:: FOX.ff.lj_param
.. autosummary::
    estimate_lj
    get_free_energy

API
---
.. autofunction:: estimate_lj
.. autofunction:: get_free_energy

"""

import reprlib
from typing import TypeVar, Optional, Mapping, MutableSequence, Dict, Generator

import numpy as np
import pandas as pd
from scipy import constants

from scm.plams import Units

__all__ = ['estimate_lj', 'get_free_energy']

A = TypeVar('A', pd.DataFrame, pd.Series, np.ndarray)


def get_free_energy(distribution: A, temperature: float = 298.15,
                    unit: str = 'kcal/mol', inf_replace: Optional[float] = np.nan) -> A:
    r"""Convert a distribution function into a free energy function.

    Given a distribution function :math:`g(r)`, the free energy
    :math:`F(g(r))` can be retrieved using a Boltzmann inversion:

    .. math::

        F(g(r)) = -RT * \text{ln} (g(r))

    Two examples of valid distribution functions would be the
    radial-  and angular distribution functions.

    .. _`scm.plams.units`: https://www.scm.com/doc/plams/components/utils.html#scm.plams.tools.units.Units

    Parameters
    ----------
    distribution : array-like
        A distribution function (*e.g.* an RDF) as an array-like object.

    temperature : :class:`float`
        The temperature in Kelvin.

    inf_replace : :class:`float`, optional
        A value used for replacing all instances of infinity (``np.inf``).

    unit : :class:`str`
        The to-be returned unit.
        See `scm.plams.Units`_ for a comprehensive overview of all allowed values.

    Returns
    -------
    :class:`pandas.DataFrame`:
        An array-like object with a free-energy function (kj/mol) of **distribution**.

    See Also
    --------
    :meth:`.MultiMolecule.init_rdf`
        Initialize the calculation of radial distribution functions (RDFs).

    :meth:`.MultiMolecule.init_adf`
        Initialize the calculation of distance-weighted angular distribution functions (ADFs).

    """  # noqa
    RT = (constants.R / 1000) * temperature  # kj/mol

    with np.errstate(divide='ignore'):
        ret = -RT * np.log(distribution)
    if inf_replace is not None:
        ret[ret == np.inf] = inf_replace

    ret *= Units.conversion_ratio('kj/mol', unit)
    return ret


def estimate_lj(rdf: pd.DataFrame, temperature: float = 298.15,
                sigma_estimate: str = 'base') -> pd.DataFrame:
    r"""Estimate the Lennard-Jones :math:`\sigma` and :math:`\varepsilon` parameters using an RDF.

    Given a radius :math:`r`, the Lennard-Jones potential :math:`V_{LJ}(r)` is defined as
    following:

    .. math::

        V_{LJ}(r) = 4 \varepsilon
        \left(
            \left(
                \frac{\sigma}{r}
            \right )^{12} -
            \left(
                \frac{\sigma}{r}
            \right )^6
        \right )

    The :math:`\sigma` and :math:`\varepsilon` parameters are estimated as following:

    * :math:`\sigma`: The radii at which the first inflection point or peak base occurs in **rdf**.
    * :math:`\varepsilon`: The minimum value in of the **rdf** ree energy multiplied by :math:`-1`.
    * All values are calculated per atom pair specified in **rdf**.

    Parameters
    ----------
    rdf : :class:`pandas.DataFrame`
        A radial distribution function.
        The columns should consist of atom-pairs.

    temperature : :class:`float`
        The temperature in Kelvin.

    sigma_estimate : :class:`str`
        Whether :math:`\sigma` should be estimated based on the base of the first peak or
        its inflection point.
        Accepted values are ``"base"`` and ``"inflection"``, respectively.

    Returns
    -------
    :class:`pandas.DataFrame`
        A Pandas DataFrame with two columns, ``"sigma"`` (Angstrom)
        and ``"epsilon"`` (kcal/mol), holding the Lennard-Jones parameters.
        Atom-pairs from **rdf** are used as index.

    See Also
    --------
    :meth:`.MultiMolecule.init_rdf`
        Initialize the calculation of radial distribution functions (RDFs).

    :func:`get_free_energy`
        Convert a distribution function into a free energy function.

    """
    G = get_free_energy(rdf, temperature)
    if sigma_estimate not in {'base', 'inflection'}:
        raise ValueError("'sigma_estimate' expected either 'base' or 'inflection'; observed value: "
                         f"{reprlib.repr(sigma_estimate)}")

    # Prepare the parameter sigma
    lj_dict: Dict[str, MutableSequence[float]] = {}
    lj_dict['epsilon'] = -1 * G.min()
    lj_dict['sigma'] = []
    sigma_append = lj_dict['sigma'].append

    for _, distr in rdf.items():
        if sigma_estimate == 'inflection':
            grad = np.gradient(distr)
            i = np.argmax(grad)  # The first inflection point in the RDF
        elif sigma_estimate == 'base':
            distr_ar = distr.values
            j = distr_ar.argmax()
            i = np.where(distr_ar[:j] <= 10**-8)[0][-1]  # Find the base of the first peak
        sigma_append(distr.index[i])

    return pd.DataFrame(lj_dict, index=G.columns)


def _charge_test(rdf: pd.DataFrame, charge_dict: Mapping[str, float],
                 temperature: float = 298.15) -> pd.Series:
    def iter_epsilon(G: pd.DataFrame) -> Generator[float, None, None]:
        for at_pair in G:
            at1, at2 = at_pair.split()
            q1q2 = charge_dict.get(at1, 0.0) * charge_dict.get(at2, 0.0)
            r_min = G[at_pair].idxmin()
            E_min = G.at[r_min, at_pair]
            yield -1 * (E_min - q1q2 / r_min)

    G = get_free_energy(rdf, temperature)
    G *= Units.conversion_ratio('kj/mol', 'au')
    G.index *= Units.conversion_ratio('Angstrom', 'Bohr')

    # Prepare the paramater epsilon; correct for Coulombic interaction if necessary
    ret = np.fromiter(iter_epsilon(G), count=G.shape[1], dtype=float)
    ret *= Units.conversion_ratio('au', 'kj/mol')
    return pd.Series(ret, index=G.columns, name='epsilon (kj/mol)')
