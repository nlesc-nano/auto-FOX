"""
FOX.functions.lj_calculate
==========================

A module for calculating non-bonded interactions using Coulomb + Lennard-Jones potentials.

"""

import textwrap
from types import MappingProxyType
from typing import Mapping, Tuple, Sequence, Optional, Iterable, Union, Dict, List
from itertools import combinations_with_replacement
from collections import abc

import numpy as np
import pandas as pd

from scm.plams import Units, Settings

from .utils import read_rtf_file, fill_diagonal_blocks, group_by_values
from ..classes.multi_mol import MultiMolecule
from ..io.read_psf import PSFContainer
from ..io.read_prm import PRMContainer

__all__ = ['get_non_bonded', 'get_V']

SliceMapping = Mapping[Tuple[str, str], Tuple[Sequence[int], Sequence[int]]]
PrmMapping = Mapping[Tuple[str, str], Tuple[float, float, float]]


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
            self.iloc[:, :] = item

    def __repr__(self) -> str:
        ret = super().__repr__()
        indent = 4 * ' '
        return f'{self.__class__.__name__}(\n{textwrap.indent(ret, indent)}\n)'

    @property
    def _constructor_expanddim(self) -> 'LJDataFrame':
        """Construct a :class:`.LJDataFrame` instance."""
        def _df(*args, **kwargs) -> LJDataFrame:
            return LJDataFrame(*args, **kwargs).__finalize__(self)
        return _df

    #: Map CP2K units to PLAMS units (see :class:`scm.plams.Units`).
    UNIT_MAPPING: Mapping[str, str] = MappingProxyType({'kcalmol': 'kcal/mol', 'kjmol': 'kj/mol'})

    def overlay_cp2k_settings(self: pd.DataFrame, cp2k_settings: Mapping) -> None:
        r"""Overlay **df** with all :math:`q`, :math:`\sigma` and :math:`\varepsilon` values from **cp2k_settings**."""  # noqa
        charge = cp2k_settings['input']['force_eval']['mm']['forcefield']['charge']
        charge_dict = {block['atom']: block['charge'] for block in charge}

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
                unit_sigma = unit_sigma[1:-1]
                unit_sigma = self.UNIT_MAPPING.get(unit_sigma, unit_sigma)
                sigma = float(sigma)

                try:
                    unit_eps, epsilon = block['epsilon'].split()
                except ValueError:
                    unit_eps, epsilon = '[kcalmol]', block['sigma']
                unit_eps = unit_eps[1:-1]
                unit_eps = self.UNIT_MAPPING.get(unit_eps, unit_eps)
                epsilon = float(epsilon)

            sigma_s[unit_sigma][atoms] = sigma
            epsilon_s[unit_eps][atoms] = epsilon

        self.set_charge(charge_dict)
        for unit, dct in epsilon_s.items():
            self.set_epsilon_pairs(dct, unit=unit)
        for unit, dct in sigma_s.items():
            self.set_sigma_pairs(dct, unit=unit)

    def overlay_prm(self, prm: Union[str, PRMContainer]) -> None:
        r"""Overlay **df** with all :math:`\sigma` and :math:`\varepsilon` values from **prm**."""
        if not isinstance(prm, PRMContainer):
            prm = PRMContainer.read(prm)

        nonbonded = prm.nonbonded.set_index(0)
        epsilon = nonbonded[2].astype(float, copy=False)
        sigma = nonbonded[3].astype(float, copy=False)
        self.set_epsilon(epsilon, unit='kcal/mol')
        self.set_sigma(sigma, unit='angstrom')

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


def get_non_bonded(mol: Union[str, MultiMolecule],
                   psf: Union[str, PSFContainer],
                   prm: Union[None, str, PRMContainer] = None,
                   rtf: Optional[str] = None,
                   cp2k_settings: Optional[Mapping] = None) -> pd.DataFrame:
    r"""Collect forcefield parameters and calculate all non-covalent interactions in **mol**.

    Forcefield parameters (*i.e.* charges and Lennard-Jones :math:`\sigma` and
    :math:`\varepsilon` values) are collected from the provided **psf**, **prm**, **rtf** and/or
    **cp2k_settings** parameters.
    Note that only that only **psf** is strictly required, though due to the lack of
    :math:`\sigma` and :math:`\varepsilon` values the resulting potential energy
    will be exclusivelly electrostatic in nature.
    Supplying a **prm** and/or **rtf** file is redundant if
    all parameters are specified in **cp2k_settings**.

    Intra-ligand interactions are ignored.

    Parameters
    ----------
    mol : :class:`str` or :class:`MultiMolecule`
        A MultiMolecule instance or the path+filename of an .xyz file.

    psf : :class:`str` or :class:`PSFContainer`
        A PSFContainer instance or the path+filename of a .psf file.
         Used for setting :math:`q` and creating atom-subsets.

    prm : :class:`str` or :class:`PRMContainer`, optional
        A PRMContainer instance or the path+filename of a .prm file.
        Used for setting :math:`\sigma` and :math:`\varepsilon`.

    rtf : :class:`str`, optional
        The path+filename of an .rtf file.
        Used for setting :math:`q`.

    cp2k_settings : :class:`Settings`, optional
        CP2K input settings.
        Used for setting :math:`q`, :math:`\sigma` and :math:`\varepsilon`.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame with the electrostatic and Lennard-Jones components of the
        (inter-ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types and
        averaged over all molecules within **mol**.
        Units are in atomic units.

    See Also
    --------
    :func:`get_V`
        Calculate all non-covalent interactions averaged over all molecules in **mol**.

    """
    psf = PSFContainer.read(psf)

    if not isinstance(mol, MultiMolecule):
        mol = MultiMolecule.from_xyz(mol)
    mol.atoms = mol_atoms = psf_to_atom_dict(psf)

    prm_df = LJDataFrame(index=mol_atoms.keys())
    prm_df.overlay_psf(psf)
    if prm is not None:
        prm_df.overlay_prm(prm)
    if cp2k_settings is not None:
        prm_df.overlay_cp2k_settings(cp2k_settings)

    slice_dict = {(i, j): (mol_atoms[i], mol_atoms[j]) for i, j in prm_df.index}
    core_atoms = set(psf.atom_type[psf.residue_id == 1])
    ligand_count = psf.residue_id.max() - 1
    return get_V(mol, slice_dict, prm_df.loc, ligand_count, core_atoms=core_atoms)


def get_V(mol: MultiMolecule, slice_mapping: SliceMapping,
          prm_mapping: PrmMapping, ligand_count: int,
          core_atoms: Optional[Iterable[str]] = None) -> pd.DataFrame:
    r"""Calculate all non-covalent interactions averaged over all molecules in **mol**.

    Parameters
    ----------
    mol : :class:`MultiMolecule`
        A MultiMolecule instance.

    slice_mapping : :class:`dict`
        A mapping of atoms-pairs to matching atomic indices.

    prm_mapping : :class:`dict`
        A mapping of atoms-pairs to matching (pair-wise) values for :math:`q`,
        :math:`sigma` and :math:`\varepsilon`.
        Units should be in atomic units.

    ligand_count : :class:`int`
        The number of ligands.

    core_atoms : :class:`set` [:class:`str`], optional
        A set of all atoms within the core.

    Returns
    -------
    :class:`pandas.DataFrame`
        A DataFrame with the electrostatic and Lennard-Jones components of the
        (inter-ligand) potential energy per atom-pair.
        The potential energy is summed over atoms with matching atom types and
        averaged over all molecules within **mol**.
        Units are in atomic units.

    """
    core_atoms = set(core_atoms) if core_atoms is not None else set()
    mol = mol * Units.conversion_ratio('Angstrom', 'au')

    df = pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples(sorted(slice_mapping.keys())),
        columns=pd.Index(['elstat', 'lj'], name='au')
    )

    for atoms, ij in slice_mapping.items():
        dist = mol.get_dist_mat(atom_subset=ij)
        if not core_atoms.intersection(atoms):
            i = len(ij[0]) // ligand_count
            j = len(ij[1]) // ligand_count
            fill_diagonal_blocks(dist, i, j)  # Set intra-ligand interactions to np.nan
        else:
            dist[dist == 0.0] = np.nan

        charge, epsilon, sigma = prm_mapping[atoms]
        df.at[atoms, 'elstat'] = get_V_elstat(charge, dist)
        df.at[atoms, 'lj'] = get_V_lj(sigma, epsilon, dist)

        if atoms[0] == atoms[1]:  # Avoid double-counting
            df.loc[atoms] /= 2

    df /= len(mol)
    return df


def get_V_elstat(q: float, dist: np.ndarray) -> float:
    r"""Calculate and sum the electrostatic potential energy given a distance matrix **dist**..

    .. math::

        V_{elstat} = \sum_{i, j} \frac{1}{4 \pi \epsilon_{0}} \frac{q_{ij}}{r_{ij}}

        q_{ij} = q_{i} * q_{j}

        4 \pi \epsilon_{0} = 1

    Parameters
    ----------
    q : :class:`float`
        The product of two charges :math:`q_{ij}`.
    dist : :class:`numpy.ndarray`
        The distance matrix :math:`r_{ij}`.
        Units should be in Bohr.

    Returns
    -------
    :class:`float`
        The elctrostatic potential energy summed over all distance pairs in **dist**.

    """
    return np.nansum(q / dist)


def get_V_lj(sigma: float, epsilon: float, dist: np.ndarray) -> float:
    r"""Calculate and sum the Lennard-Jones potential given a distance matrix **dist**.

    .. math::

        V_{LJ} = \sum_{i, j} 4 \varepsilon_{ij}
        \left(
            \left(
                \frac{\sigma_{ij}}{r_{ij}}
            \right )^{12} -
            \left(
                \frac{\sigma_{ij}}{r_{ij}}
            \right )^6
        \right )

        \sigma_{ij} = \frac{\sigma_{i} \sigma_{j}}{2}

        \varepsilon_{ij} = \sqrt{\varepsilon_{i} \varepsilon_{j}}

    Parameters
    ----------
    sigma : :class:`float`
        The arithmetic mean of two :math:`\sigma` parameters: :math:`\sigma_{ij}`.
        Units should be in Bohr.
    epsilon : :class:`float`
        The geometric mean of two :math:`\varepsilon` parameters: :math:`\varepsilon_{ij}`.
        Units should be in Hartree.
    dist : :class:`numpy.ndarray`
        The distance matrix :math:`r_{ij}`.
        Units should be in Bohr.

    Returns
    -------
    :class:`float`
        The Lennard-Jones potential energy summed over all distance pairs in **dist**.

    """
    sigma_dist = (sigma / dist)**6
    lj = sigma_dist**2 - sigma_dist
    lj *= epsilon * 4
    return np.nansum(lj)


def psf_to_atom_dict(psf: Union[str, PSFContainer]) -> Dict[str, List[int]]:
    """Create a new dictionary of atoms and their respective indices.

    Parameters
    ----------
    psf : :class:`str` or :class:`PSFContainer`
        A PSFContainer instance or a file-like object representing a .psf file.

    Returns
    -------
    :class:`dict` [:class:`str`, :class:`list` [:class:`int`]]
        A dictionary with atom types as keys and lists of matching atomic indices as values.
        The indices are 0-based.

    """
    if not isinstance(psf, PSFContainer):
        psf = PSFContainer.read(psf)

    try:
        iterator = enumerate(psf.atom_type)
    except AttributeError as ex:
        err = "The 'psf' parameter is of invalid type: '{psf.__class__.__name__}'"
        raise TypeError(err).with_traceback(ex.__traceback__)
    return group_by_values(iterator)
