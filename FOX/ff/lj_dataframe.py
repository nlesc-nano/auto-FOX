import textwrap
from types import MappingProxyType
from typing import Union, Iterable, Mapping, Dict, Tuple, Callable
from itertools import combinations_with_replacement
from collections import abc

import numpy as np
import pandas as pd

from scm.plams import Settings, Units

from ..io.read_prm import PRMContainer
from ..io.read_psf import PSFContainer
from ..functions.utils import read_rtf_file

__all__ = ['LJDataFrame']


class LJDataFrame(pd.DataFrame):

    def __init__(self, data: Union[None, float, Iterable] = None,
                 index: Iterable[str] = None,
                 columns: None = None,
                 dtype: Union[None, str, type, np.dtype] = None,
                 copy: bool = False) -> None:
        """Initialize a :class:`LJDataFrame` instance."""
        if index is None:
            raise TypeError("The 'index' parameter expects an iterable of atom types; "
                            f"observed type: '{index.__class__.__name__}'")
        if columns is not None:
            raise TypeError("The 'columns' parameter should be 'None'")

        # Create the DataFrame
        index = pd.MultiIndex.from_tuples(combinations_with_replacement(sorted(index), 2))
        columns = ['charge', 'epsilon', 'sigma']
        super().__init__(0.0, index=index, columns=columns)

        if isinstance(data, abc.Mapping):
            for k, v in data.items():
                if k not in columns:
                    raise KeyError(f"Invalid key {repr(k)}; allowed keys: "
                                   "'charge', 'epsilon' and 'sigma'")
                self[k] = v

        elif isinstance(data, abc.Iterable):
            for i, item in enumerate(data):
                self.iloc[:, i] = item

        elif data is not None:
            self.iloc[:, :] = data

    def __repr__(self) -> str:
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

    def overlay_prm(self, prm: Union[str, PRMContainer]) -> None:
        r"""Overlay **df** with all :math:`\sigma` and :math:`\varepsilon` values from **prm**."""
        if not isinstance(prm, PRMContainer):
            prm = PRMContainer.read(prm)

        nonbonded = prm.nonbonded
        if nonbonded is None:
            return None

        epsilon = nonbonded[2]
        sigma = nonbonded[3]
        self.set_epsilon(epsilon, unit='kcal/mol')
        self.set_sigma(sigma, unit='angstrom')

        # Check if non-bonded pairs have been explicitly specified in the ``nbfix`` block
        nbfix = prm.nbfix
        if nbfix is None:
            return

        is_null = nbfix[[2, 3]].isnull().any(axis=1)
        epsilon_pair = nbfix.loc[~is_null, 2]
        sigma_pair = nbfix.loc[~is_null, 3]

        self.set_epsilon_pairs(epsilon_pair, unit='kcal/mol')
        self.set_sigma_pairs(sigma_pair, unit='angstrom')

    def overlay_rtf(self, rtf: str) -> None:
        r"""Overlay **df** with all :math:`q` values from **rtf**."""
        charge_dict: Dict[str, float] = dict(zip(*read_rtf_file(rtf)))
        self.set_charge(charge_dict)

    def overlay_psf(self, psf: Union[str, PRMContainer]) -> None:
        r"""Overlay **df** with all :math:`q` values from **psf**."""
        if not isinstance(psf, PSFContainer):
            psf = PSFContainer.read(psf)

        charge = psf.atoms.set_index('atom type')['charge']
        charge_dict = charge.to_dict()
        self.set_charge(charge_dict)

    def set_charge(self, charge_mapping: Mapping[str, float]) -> None:
        """Set :math:`q_{i} * q_{j}`."""
        atom_pairs = combinations_with_replacement(sorted(charge_mapping.keys()), 2)
        for i, j in atom_pairs:
            charge = charge_mapping[i] * charge_mapping[j]
            self.at[(i, j), 'charge'] = charge

    def set_epsilon(self, epsilon_mapping: Mapping[str, float], unit: str = 'kj/mol') -> None:
        r"""Set :math:`\sqrt{\varepsilon_{i} * \varepsilon_{j}}`."""
        atom_pairs = combinations_with_replacement(sorted(epsilon_mapping.keys()), 2)
        for i, j in atom_pairs:
            epsilon = (epsilon_mapping[i] * epsilon_mapping[j])**0.5
            epsilon *= Units.conversion_ratio(unit, 'au')
            self.at[(i, j), 'epsilon'] = epsilon

    def set_sigma(self, sigma_mapping: Mapping[str, float],
                  unit: str = 'nm') -> None:
        r"""Set :math:`\frac{ \sigma_{i} * \sigma_{j} }{2}`."""
        unit2au = Units.conversion_ratio(unit, 'au')
        atom_pairs = combinations_with_replacement(sorted(sigma_mapping.keys()), 2)
        for i, j in atom_pairs:
            sigma = (sigma_mapping[i] + sigma_mapping[j]) / 2
            sigma *= unit2au
            self.at[(i, j), 'sigma'] = sigma

    def set_charge_pairs(self, charge_mapping: Mapping[Tuple[str, str], float]) -> None:
        """Set :math:`q_{ij}`."""
        for _ij, charge in charge_mapping.items():
            ij = tuple(sorted(_ij))
            self.at[ij, 'charge'] = charge

    def set_epsilon_pairs(self, epsilon_mapping: Mapping[Tuple[str, str], float],
                          unit: str = 'kj/mol') -> None:
        r"""Set :math:`\varepsilon_{ij}`."""
        unit2au = Units.conversion_ratio(unit, 'au')
        for _ij, epsilon in epsilon_mapping.items():
            ij = tuple(sorted(_ij))
            epsilon *= unit2au
            self.at[ij, 'epsilon'] = epsilon

    def set_sigma_pairs(self, sigma_mapping: Mapping[Tuple[str, str], float],
                        unit: str = 'nm') -> None:
        r"""Set :math:`\sigma_{ij}`."""
        unit2au = Units.conversion_ratio(unit, 'au')
        for _ij, sigma in sigma_mapping.items():
            ij = tuple(sorted(_ij))
            sigma *= unit2au
            self.at[ij, 'sigma'] = sigma
