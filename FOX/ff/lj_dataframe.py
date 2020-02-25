"""
FOX.ff.lj_dataframe
===================

A module for holding the :class:`LJDataFrame` class.

Index
-----
.. currentmodule:: FOX.ff.lj_dataframe
.. autosummary::
    LJDataFrame

API
---
.. autoclass:: LJDataFrame
    :members:

"""

import textwrap
from types import MappingProxyType
from typing import Union, Iterable, Mapping, Dict, Tuple, Callable, Optional
from itertools import combinations_with_replacement, chain, product
from collections import abc

import numpy as np
import pandas as pd

from scm.plams import Settings, Units

from ..io.read_prm import PRMContainer
from ..io.read_psf import PSFContainer
from ..functions.utils import read_rtf_file

__all__ = ['LJDataFrame']


def gmean(x: Union[float, Iterable[float]]) -> float:
    r"""Return the geometric mean of **x**.

    .. math::

        \left(
            \prod_{i=0}^{n} x_{i}
        \right)^{\frac{1}{n}}
        \quad \text{with} \quad \boldsymbol{x} \in \mathbb{R}^{n}

    """
    try:
        ar = np.array(x, dtype=float, ndmin=1, copy=False)
    except TypeError as ex:
        if isinstance(x, abc.Iterable):
            ar = np.fromiter(x, dtype=float)
        else:
            raise ex

    power = 1 / len(ar)
    return ar.prod()**power


class LJDataFrame(pd.DataFrame):
    """A subclass of :class:`pandas.DataFrame` aimed at holding forcefield parameters."""

    def __init__(self, data: Union[None, float, Iterable] = None,
                 index: Iterable[str] = None,
                 columns: None = None,
                 dtype: Union[None, str, type, np.dtype] = None,
                 copy: bool = False) -> None:
        """Initialize a :class:`LJDataFrame` instance."""
        # Validate the index and columns
        if index is None:
            raise TypeError("The 'index' parameter expects an iterable of atom types; "
                            f"observed type: '{index.__class__.__name__}'")
        if columns is not None:
            raise TypeError("The 'columns' parameter should be 'None'")

        # Create the DataFrame
        index = pd.MultiIndex.from_tuples(combinations_with_replacement(sorted(index), 2))
        super().__init__(0.0, index=index, columns=['charge', 'epsilon', 'sigma'])

        if callable(getattr(data, 'items', None)):
            columns = set(self.columns)
            for k, v in data.items():
                if k not in columns:
                    raise KeyError(f"Invalid key {k!r}; allowed keys: {tuple(columns)}")
                self[k] = v

        elif isinstance(data, abc.Iterable):
            for i, item in enumerate(data):
                self.iloc[:, i] = item

        elif data is not None:
            self.iloc[:, :] = data

    def __repr__(self) -> str:
        """Return a string-representation of this instance."""
        ret = super().__repr__()
        indent = 4 * ' '
        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent)}\n)'

    @property
    def _constructor_expanddim(self) -> Callable:
        """Construct a :class:`.LJDataFrame` instance."""
        def _df(*args, **kwargs) -> 'LJDataFrame':
            return LJDataFrame(*args, **kwargs).__finalize__(self)
        return _df

    #: Map CP2K units to PLAMS units (see :class:`scm.plams.Units`).
    UNIT_MAPPING: Mapping[str, str] = MappingProxyType({'kcalmol': 'kcal/mol', 'kjmol': 'kj/mol'})

    def overlay_cp2k_settings(self: pd.DataFrame, cp2k_settings: Mapping) -> None:
        r"""Overlay **df** with all :math:`q`, :math:`\sigma` and :math:`\varepsilon` values from **cp2k_settings**."""  # noqa
        charge = cp2k_settings['input']['force_eval']['mm']['forcefield']['charge']
        charge_dict = {block['atom']: float(block['charge']) for block in charge}

        lj = cp2k_settings['input']['force_eval']['mm']['forcefield']['nonbonded']['lennard-jones']  # noqa
        epsilon_s = Settings()
        sigma_s = Settings()
        for block in lj:
            with Settings.supress_missing():
                atoms = tuple(block['atoms'].split())

                try:
                    unit_sigma, sigma = block['sigma'].split()
                except ValueError:
                    unit_sigma, sigma = '[angstrom]', block['sigma']
                except (TypeError, KeyError):
                    unit_sigma = sigma = None

                try:
                    unit_eps, epsilon = block['epsilon'].split()
                except ValueError:
                    unit_eps, epsilon = '[kcalmol]', block['sigma']
                except (TypeError, KeyError):
                    unit_eps = epsilon = None

            if sigma is not None:
                unit_sigma = unit_sigma[1:-1]
                unit_sigma = self.UNIT_MAPPING.get(unit_sigma, unit_sigma)
                sigma_s[unit_sigma][atoms] = float(sigma)

            if epsilon is not None:
                unit_eps = unit_eps[1:-1]
                unit_eps = self.UNIT_MAPPING.get(unit_eps, unit_eps)
                epsilon_s[unit_eps][atoms] = float(epsilon)

        self.set_charge(charge_dict)
        for unit, dct in epsilon_s.items():
            self.set_epsilon_pairs(dct, unit=unit)
        for unit, dct in sigma_s.items():
            self.set_sigma_pairs(dct, unit=unit)

    def overlay_prm(self, prm: Union[str, PRMContainer], pairs14: bool = False) -> None:
        r"""Overlay **df** with all :math:`\sigma` and :math:`\varepsilon` values from **prm**."""
        # In the .prm format nonbonded parameters are stored in columns 2 & 3
        # Explicit 1,4-nonbonded parameters are stored in columns 4 & 5
        i, j = (2, 3) if not pairs14 else (4, 5)
        if not isinstance(prm, PRMContainer):
            prm = PRMContainer.read(prm)

        nonbonded = prm.nonbonded
        if nonbonded is None:
            return None

        epsilon = nonbonded[i]
        sigma = nonbonded[j] * 2  # The .prm format stores sigma / 2, not sigma
        sigma /= 2**(1/6)  # Convert r_min to sigma
        self.set_epsilon(epsilon, unit='kcal/mol')
        self.set_sigma(sigma, unit='angstrom')

        # Check if certain non-bonded pairs have been explicitly specified in the ``nbfix`` block
        nbfix = prm.nbfix
        if nbfix is None:
            return

        is_null = nbfix[[i, j]].isnull().any(axis=1)
        epsilon_pair = nbfix.loc[~is_null, i]
        sigma_pair = nbfix.loc[~is_null, j] * 2  # The .prm format stores sigma / 2, not sigma
        sigma_pair /= 2**(1/6)  # Convert r_min to sigma

        self.set_epsilon_pairs(epsilon_pair, unit='kcal/mol')
        self.set_sigma_pairs(sigma_pair, unit='angstrom')

    def overlay_rtf(self, rtf: str) -> None:
        r"""Overlay **df** with all :math:`q` values from **rtf**."""
        charge_dict: Dict[str, float] = dict(zip(*read_rtf_file(rtf)))
        self.set_charge(charge_dict)

    def overlay_psf(self, psf: Union[str, PSFContainer]) -> None:
        r"""Overlay **df** with all :math:`q` values from **psf**."""
        if not isinstance(psf, PSFContainer):
            psf = PSFContainer.read(psf)

        charge = psf.atoms.set_index('atom type')['charge']
        charge_dict = self._update_atom_type(charge, psf)
        self.set_charge(charge_dict)

    def _update_atom_type(self, series: pd.Series, psf: PSFContainer) -> Dict[str, float]:
        """Update the atom types of all atom types with multiple non-unique charges."""
        # Evaluate the Series for duplicate key/value pairs
        idx, data = zip(*{kv for kv in series.items()})
        series_unique = pd.Series(data, index=idx, name=series.name)
        series_unique = series_unique.loc[np.isin(series_unique.index,
                                                  list(set(chain.from_iterable(self.index))))]
        is_unique = ~series_unique.index.duplicated()
        if is_unique.all():
            return series_unique.to_dict()

        # Rename duplicate elements within index
        idx_update = '_' + series_unique.groupby(level=0).cumcount().astype(str)
        series_unique.index += idx_update.replace('_0', '')

        # Add new indices to the existing DataFrame
        idx_set = set(chain.from_iterable(self.index))
        idx_diff = set(series_unique.index).difference(chain.from_iterable(self.index))

        for i, j in product(idx_set, idx_diff):
            ij = tuple(sorted([i, j]))
            ij_0 = tuple(sorted((i, j.split('_')[0])))
            self.loc[ij, :] = self.loc[ij_0, :]

        for i, j in product(idx_diff, idx_diff):
            ij = tuple(sorted([i, j]))
            ij_0 = tuple(sorted((i.split('_')[0], j.split('_')[0])))
            self.loc[ij, :] = self.loc[ij_0, :]

        self.sort_index(inplace=True)

        # Update atom types in the PSFContainer
        with pd.option_context('mode.chained_assignment', None):
            for k2, v in series_unique.items():
                if k2 in series:
                    continue
                k1 = k2.split('_')[0]
                psf.atom_type[(psf.atom_type == k1) & (psf.charge == v)] = k2

        return series_unique.to_dict()

    def _set_prm(self, atom_mapping: Mapping[str, float], key: str,
                 func: Callable[[Tuple[float, float]], float],
                 unit: Optional[str] = None) -> None:
        unit2au = 1 if unit is None else Units.conversion_ratio(unit, 'au')
        atom_pairs = combinations_with_replacement(sorted(atom_mapping.keys()), 2)
        for at1, at2 in atom_pairs:
            value = func([atom_mapping[at1], atom_mapping[at2]])
            value *= unit2au
            self.at[(at1, at2), key] = value

    def set_charge(self, charge_mapping: Mapping[str, float]) -> None:
        """Set :math:`q_{i} * q_{j}`."""
        self._set_prm(charge_mapping, 'charge', func=np.product, unit=None)

    def set_epsilon(self, epsilon_mapping: Mapping[str, float], unit: str = 'kj/mol') -> None:
        r"""Set :math:`\sqrt{\varepsilon_{i} * \varepsilon_{j}}`."""
        self._set_prm(epsilon_mapping, 'epsilon', func=gmean, unit=unit)

    def set_sigma(self, sigma_mapping: Mapping[str, float], unit: str = 'nm') -> None:
        r"""Set :math:`\frac{ \sigma_{i} + \sigma_{j} }{2}`."""
        self._set_prm(sigma_mapping, 'sigma', func=np.mean, unit=unit)

    def _set_prm_pairs(self, atom_pair_mapping: Mapping[Tuple[str, str], float],
                       key: str, unit: Optional[str] = None) -> None:
        unit2au = 1 if unit is None else Units.conversion_ratio(unit, 'au')
        for _at_tup, value in atom_pair_mapping.items():
            at_tup = tuple(sorted(_at_tup))
            value *= unit2au
            self.at[at_tup, key] = value

    def set_charge_pairs(self, charge_mapping: Mapping[Tuple[str, str], float]) -> None:
        """Set :math:`q_{ij}`."""
        self._set_prm_pairs(charge_mapping, 'charge', unit=None)

    def set_epsilon_pairs(self, epsilon_mapping: Mapping[Tuple[str, str], float],
                          unit: str = 'kj/mol') -> None:
        r"""Set :math:`\varepsilon_{ij}`."""
        self._set_prm_pairs(epsilon_mapping, 'epsilon', unit=unit)

    def set_sigma_pairs(self, sigma_mapping: Mapping[Tuple[str, str], float],
                        unit: str = 'nm') -> None:
        r"""Set :math:`\sigma_{ij}`."""
        self._set_prm_pairs(sigma_mapping, 'sigma', unit=unit)
