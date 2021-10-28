"""A Module for the :class:`MultiMolecule` class.

Index
-----
.. currentmodule:: FOX
.. autosummary::
    MultiMolecule

API
---
.. autoclass:: FOX.MultiMolecule
    :members:
    :noindex:

"""

from __future__ import annotations

import copy
import warnings
from os import PathLike
from collections import abc, defaultdict
from itertools import (
    chain, combinations_with_replacement, zip_longest, islice, repeat, permutations
)
from typing import (
    Sequence, Optional, Union, List, Hashable, Callable, Iterable, Dict, Tuple, Any, Mapping,
    overload, TypeVar, Type, Container, cast, TYPE_CHECKING, Sized, Iterator, NoReturn
)

import numpy as np
import pandas as pd
from scipy import constants
from scipy.fftpack import fft
from scipy.spatial.distance import cdist

from scm.plams import Molecule, Atom, Bond, PeriodicTable
from nanoutils import group_by_values, Literal

from ..utils import slice_iter, lattice_to_volume
from .multi_mol_magic import _MultiMolecule, AliasTuple
from ..io.read_kf import read_kf
from ..io.read_xyz import read_multi_xyz
from ..functions.rdf import get_rdf, get_rdf_df
from ..functions.adf import get_adf_df, _adf_inner_cdktree, _adf_inner
from ..functions.molecule_utils import fix_bond_orders, separate_mod
from ..functions.periodic import parse_periodic

if TYPE_CHECKING:
    import numpy.typing as npt

try:
    import dask
    DASK_EX: Optional[Exception] = None
except Exception as ex:
    DASK_EX = ex
    _warn = ImportWarning(str(ex))
    _warn.__cause__ = ex
    warnings.warn(_warn)
    del _warn

try:
    from ase import Atoms
    ASE_EX: Optional[ImportError] = None
except ImportError as ex:
    ASE_EX = ex

__all__ = ['MultiMolecule']

MT = TypeVar('MT', bound='MultiMolecule')
_DType = TypeVar("_DType", bound=np.dtype)
_T = TypeVar("_T")

MolSubset = Union[None, slice, int]
AtomSubset = Union[
    None, slice, range, int, str, Sequence[int], Sequence[str], Sequence[Sequence[int]]
]


def neg_exp(x: np.ndarray) -> np.ndarray:
    """Return :math:`e^{-x}`."""
    return np.exp(-x)


class _GetNone:
    def __getitem__(self, key: object) -> None:
        return None


def _parse_atom_pairs(
    mol: MultiMolecule,
    atom_pairs: Iterable[tuple[str, str]],
) -> dict[str, list[npt.NDArray[np.intp]]]:
    """Helper function for translating atom-pairs into a dict of indice-array-pairs."""
    pair_dict = {}
    for atoms in atom_pairs:
        key = " ".join(atoms)

        idx_list = []
        try:
            for at in atoms:
                idx = mol.atoms.get(at)
                if idx is None:
                    super_at, slc = mol.atoms_alias[at]
                    idx = mol.atoms[super_at][slc]
                idx_list.append(idx)
        except KeyError as ex:
            raise ValueError(f"Unknown atom type: {ex}") from None

        pair_dict[key] = idx_list
    return pair_dict


class MultiMolecule(_MultiMolecule):
    """A class designed for handling a and manipulating large numbers of molecules.

    More specifically, different conformations of a single molecule as derived from, for example,
    an intrinsic reaction coordinate calculation (IRC) or a molecular dymanics trajectory (MD).
    The class has access to four attributes (further details are provided under parameters):

    Parameters
    ----------
    coords : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, n, 3)`
        A 3D array with the cartesian coordinates of :math:`m` molecules with :math:`n` atoms.
    atoms : :class:`dict[str, list[str]] <dict>`
        A dictionary with atomic symbols as keys and matching atomic indices as values.
        Stored in the :attr:`MultiMolecule.atoms` attribute.
    bonds : :class:`np.ndarray[np.int64] <numpy.ndarray>`, shape :math:`(k, 3)`
        A 2D array with indices of the atoms defining all :math:`k` bonds
        (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
        Stored in the :attr:`MultiMolecule.bonds` attribute.
    properties : :class:`plams.Settings <scm.plams.core.settings.Settings>`
        A Settings instance for storing miscellaneous user-defined (meta-)data.
        Is devoid of keys by default.
        Stored in the :attr:`MultiMolecule.properties` attribute.
    lattice : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, 3, 3)` or :math:`(3, 3)`, optional
        Lattice vectors for periodic systems.
        For non-periodic systems this value should be :data:`None`.

    Attributes
    ----------
    atoms : :class:`dict[str, list[str]] <dict>`
        A dictionary with atomic symbols as keys and matching atomic indices as values.
    bonds : :class:`np.ndarray[np.int64] <numpy.ndarray>`, shape :math:`(k, 3)`
        A 2D array with indices of the atoms defining all :math:`k` bonds
        (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
    properties : :class:`plams.Settings <scm.plams.core.settings.Settings>`
        A Settings instance for storing miscellaneous user-defined (meta-)data.
        Is devoid of keys by default.
    lattice : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, 3, 3)` or :math:`(3, 3)`, optional
        Lattice vectors for periodic systems.
        For non-periodic systems this value should be :data:`None`.

    """  # noqa: E501

    @overload
    def round(self: MT, decimals: int = ..., *, inplace: Literal[False] = ...) -> MT: ...  # type: ignore[misc] # noqa: E501
    @overload
    def round(self, decimals: int = ..., *, inplace: Literal[True] = ...) -> None: ...
    def round(self, decimals=0, *, inplace=False):  # noqa: E301
        """Round the Cartesian coordinates of this instance to a given number of decimals.

        Parameters
        ----------
        decimals : :class:`int`
            The number of decimals per element.
        inplace : :class:`bool`
            Instead of returning the new coordinates, perform an inplace update of this instance.

        """
        if inplace:
            self[:] = super().round(decimals)
            return None
        else:
            ret = self.copy()
            ret[:] = super().round(decimals)
            return ret

    def delete_atoms(self: MT, atom_subset: AtomSubset) -> MT:
        """Create a copy of this instance with all atoms in **atom_subset** removed.

        Parameters
        ----------
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`FOX.MultiMolecule`
            A new molecule with all atoms in **atom_subset** removed.

        Raises
        ------
        TypeError
            Raised if **atom_subset** is :data:`None`.

        """
        if atom_subset is None:
            raise TypeError("'None' is an invalid value for 'atom_subset'")

        # Define subsets
        at_subset = self._get_atom_subset(atom_subset, as_array=True)
        bool_ar = np.ones(self.shape[1], dtype=bool)
        bool_ar[at_subset] = False

        # Delete atoms
        ret = self[:, bool_ar]  # Boolean-array slicing always creates a copy
        ret.__dict__ = copy.deepcopy(self.__dict__)

        # Update :attr:`.MultiMolecule.atoms`
        symbols = self.symbol[bool_ar]
        ret.atoms = group_by_values(enumerate(symbols))

        # Update :attr:`MultiMolecule.atoms_alias`
        if len(ret.atoms_alias):
            alias_dct = {}
            for k1, (k2, idx_ar) in ret.atoms_alias.items():
                idx_ar2 = self.atoms[k2][idx_ar]
                bool_slc = bool_ar[idx_ar2]
                if not bool_slc.any():
                    pass
                elif bool_slc.all():
                    alias_dct[k1] = AliasTuple(k2, idx_ar)
                else:
                    idx_ar_new = np.arange(idx_ar2, dtype=np.intp)[bool_slc]
                    alias_dct[k1] = AliasTuple(k2, idx_ar_new)
            ret.atoms_alias = alias_dct
        return ret

    def get_supercell(self: MT, supercell_size: tuple[int, int, int]) -> MT:
        """Construct a new supercell by duplicating the molecule.

        Parameters
        ----------
        supercell_size : :class:`tuple[int, int, int]`
            The number of new unit cells along each of the three Cartesian axes.

        Returns
        -------
        :class:`FOX.MultiMolecule`
            The new supercell constructed from **self**.

        """
        if self.lattice is None:
            raise ValueError(f"Cannot construct a supercell from a {self.__class__.__name__} "
                             "without a lattice")

        # Parse and validate th
        ar = np.array(supercell_size).astype(np.int64, copy=False, casting="same_kind")
        if ar.shape != (3,):
            raise ValueError('`duplicates` expected a sequence of length 3')
        elif not (ar >= 1).all():
            raise ValueError('`duplicates` values must be larger than or equal to 1')

        mult = np.array(
            [(i, j, k) for i in range(ar[0]) for j in range(ar[1]) for k in range(ar[2])]
        )
        lat = self.lattice if self.lattice.ndim == 2 else self.lattice[None, ...]
        lat_trans = (lat[:, None, ...] * mult[None, ..., None]).sum(axis=-1)

        mol_trans = lat_trans[..., None, :] + self[:, None, ...]
        if mol_trans.shape[1] == 1:
            return mol_trans[..., 0]
        else:
            mol, other = mol_trans[..., 0], mol_trans[..., 1:]
            return mol.concatenate(other[..., 1:], lattice=lat * ar)

    def concatenate(
        self: MT,
        other: Iterable[MT],
        lattice: None | npt.ArrayLike = None,
        axis: Literal[1] = 1,
    ) -> MT:
        """Concatenate one or more molecules along the user-specified axis.

        Parameters
        ----------
        other : :class:`Iterable[FOX.MultiMolecule] <collections.abc.Iterable>`
            The to-be concatenated molecules.
        lattice : :class:`np.ndarray <numpy.ndarray>`, optional
            The lattice of the new molecule.

        Returns
        -------
        :class:`FOX.MultiMolecule`
            The newly concatenated molecule.

        """
        if axis != 1:
            raise NotImplementedError

        mol_list = [self, *other]
        if any(m.lattice is not None for m in mol_list) and lattice is None:
            raise ValueError("Cannot concatenate lattice-containing molecules without explicitly "
                             "specifying the new `lattice`")
        elif any(len(m.atoms_alias) for m in mol_list):
            raise NotImplementedError
        elif len({m.shape[0] for m in mol_list}) != 1:
            raise ValueError("All `MultiMolecule` instances must be of the same length")

        # Construct the new atoms
        proto_atoms = defaultdict(list)
        offset = 0
        for i, m in enumerate(mol_list):
            if i:
                offset += m.shape[1]
            for k, v in m.atoms.items():
                proto_atoms[k](v + offset)
        atoms = {k: np.fromiter(chain.from_iterable(v), np.int64) for k, v in proto_atoms.items()}

        # Construct the new coordinates
        coords_shape = (self.shape[0], sum(m.shape[1] for m in mol_list), 3)
        coords = np.empty(coords_shape, dtype=np.float64)
        i, j = 0, 0
        for m in mol_list:
            j += m.shape[1]
            coords[:, i:j] = m
            i += m.shape[1]

        # Construct the new bonds
        bonds_shape = (sum(m.bonds.shape[0] for m in mol_list), 3)
        bonds = np.empty(bonds_shape, dtype=np.int64)
        i, j = 0, 0
        for m in mol_list:
            j += m.bonds.shape[0]
            bonds[i:j] = m.bonds
            i += m.bonds.shape[0]

        cls = type(self)
        return cls(coords, atoms, bonds, self.properties.copy(), {}, lattice)

    def add_atoms(self: MT, coords: np.ndarray, symbols: Union[str, Iterable[str]] = 'Xx') -> MT:
        """Create a copy of this instance with all atoms in **atom_subset** appended.

        Examples
        --------
        .. code:: python

            >>> import numpy as np
            >>> from FOX import MultiMolecule, example_xyz

            >>> mol = MultiMolecule.from_xyz(example_xyz)
            >>> coords: np.ndarray = np.random.rand(73, 3)  # Add 73 new atoms with random coords
            >>> symbols = 'Br'

            >>> mol_new: MultiMolecule = mol.add_atoms(coords, symbols)

            >>> print(repr(mol))
            MultiMolecule(..., shape=(4905, 227, 3), dtype='float64')
            >>> print(repr(mol_new))
            MultiMolecule(..., shape=(4905, 300, 3), dtype='float64')

        Parameters
        ----------
        coords : array-like
            A :math:`(3,)`, :math:`(n, 3)`, :math:`(m, 3)` or :math:`(m, n, 3)` array-like object
            with :code:`m == len(self)`.
            Represents the Cartesian coordinates of the to-be added atoms.
        symbols : :class:`str` or :class:`Iterable[str] <collections.abc.Iterable>`
            One or more atomic symbols of the to-be added atoms.

        Returns
        -------
        :class:`FOX.MultiMolecule`
            A new molecule with all atoms in **atom_subset** appended.

        """
        # Reshape the passed coordinates
        coords = np.array(coords, dtype=float, ndmin=3, copy=False)
        i, j, k = coords.shape
        if i == len(self):
            coords_ = coords.reshape(len(self), 1, 3) if k == 1 else coords
        elif i == 3 and j == k == 1:
            coords_ = np.empty((len(self), 1, 3))
            coords_[:] = coords.reshape(1, 1, 3)
        else:
            coords_ = np.empty((len(self), i, 3))
            coords_[:] = coords.reshape(1, i, 3)
        j = coords_.shape[1]

        # Append
        cls = type(self)
        ret = cls(np.hstack([self, coords_]))
        ret.__dict__ = copy.deepcopy(self.__dict__)

        # Update atomic symbols & indices
        symbols = repeat(symbols, j) if isinstance(symbols, str) else islice(symbols, j)
        dct = {k: v.tolist() for k, v in ret.atoms.items()}
        atoms_append = {k: v.append for k, v in dct.items()}
        for i, item in enumerate(symbols, self.shape[1]):
            try:
                atoms_append[item](i)
            except KeyError:
                dct[item] = list_ = [i]
                atoms_append[item] = list_.append
        ret.atoms = dct
        return ret

    def guess_bonds(self, atom_subset: AtomSubset = None) -> None:
        """Guess bonds within the molecules based on atom type and inter-atomic distances.

        Bonds are guessed based on the first molecule in this instance
        Performs an inplace modification of **self.bonds**

        Parameters
        ----------
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            A tuple of atomic symbols. Bonds are guessed between all atoms
            whose atomic symbol is in **atom_subset**.
            If :data:`None`, guess bonds for all atoms in this instance.

        """
        at_subset = self._get_atom_subset(atom_subset, as_array=True)
        at_subset.sort()

        # Guess bonds
        mol = self.as_Molecule(mol_subset=0, atom_subset=atom_subset)[0]
        mol.guess_bonds()
        fix_bond_orders(mol)
        self.bonds = MultiMolecule.from_Molecule(mol, subset='bonds').bonds

        # Update indices in **self.bonds** to account for **atom_subset**
        self.atom1 = at_subset[self.atom1]
        self.atom2 = at_subset[self.atom2]
        self.bonds[:, 0:2].sort(axis=1)
        idx = self.bonds[:, 0:2].argsort(axis=0)[:, 0]
        self.bonds = self.bonds[idx]

    @overload
    def random_slice(self: MT, start: int = ..., stop: Optional[int] = ..., p: float = ..., inplace: Literal[False] = ...) -> MT: ...  # type: ignore[misc] # noqa: E501
    @overload
    def random_slice(self, start: int = ..., stop: Optional[int] = ..., p: float = ..., inplace: Literal[True] = ...) -> None: ...  # noqa: E501
    def random_slice(self, start=0, stop=None, p=0.5, inplace=False):  # noqa: E301
        """Construct a new :class:`MultiMolecule` instance by randomly slicing this instance.

        The probability of including a particular element is equivalent to **p**.

        Parameters
        ----------
        start : :class:`int`
            Start of the interval.
        stop : :class:`int`, optional
            End of the interval.
        p : :class:`float`
            The probability of including each particular molecule in this instance.
            Values must be between ``0`` (0%) and ``1`` (100%).
        inplace : :class:`bool`
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        :class:`FOX.MultiMolecule` or :data:`None`
            If **inplace** is :data:`True`, return a new molecule.

        Raises
        ------
        ValueError
            Raised if **p** is smaller than ``0.0`` or larger than ``1.0``.

        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError("The supplied probability, 'p': {:f}, must be larger "
                             "than 0.0 and smaller than 1.0".format(p))

        stop = stop or self.shape[0]
        idx_range = np.arange(start, stop)
        size = int(p * len(idx_range))
        idx = np.random.choice(idx_range, size=size, replace=False)

        if inplace:
            self[:] = self[idx]
            return None
        else:
            return self[idx].copy()

    @overload
    def reset_origin(self, mol_subset: MolSubset = ..., atom_subset: AtomSubset = ..., inplace: Literal[True] = ..., rot_ref: Optional[npt.ArrayLike] = ...) -> None: ...  # type: ignore[misc] # noqa: E501
    @overload
    def reset_origin(self: MT, mol_subset: MolSubset = ..., atom_subset: AtomSubset = ..., inplace: Literal[False] = ..., rot_ref: Optional[npt.ArrayLike] = ...) -> MT: ...  # noqa: E501
    def reset_origin(self, mol_subset=None, atom_subset=None, inplace=True, rot_ref=None):  # noqa: E301,E501
        """Reallign all molecules in this instance.

        All molecules in this instance are rotating and translating, by performing a partial partial
        Procrustes superimposition with respect to the first molecule in this instance.

        The superimposition is carried out with respect to the first molecule in this instance.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        inplace : :class:`bool`
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        :class:`FOX.MultiMolecule` or :data:`None`
            If **inplace** is :data:`True`, return a new :class:`MultiMolecule` instance.

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Remove translations
        coords = self[i, j, :] - self[i, j, :].mean(axis=1)[:, None, :]

        if rot_ref is None:
            ref_ar = coords[0]
        else:
            ref_ar = np.asarray(rot_ref)

        # Peform a singular value decomposition on the covariance matrix
        H = np.swapaxes(coords, 1, 2) @ ref_ar
        U, _, Vt = np.linalg.svd(H)
        V, Ut = np.swapaxes(Vt, 1, 2), np.swapaxes(U, 1, 2)

        # Construct the rotation matrix
        rotmat = np.ones_like(U)
        rotmat[:, 2, 2] = np.linalg.det(V @ Ut)
        rotmat *= V@Ut

        # Return or perform an inplace update of this instance
        if inplace:
            self[i, j, :] = coords @ np.swapaxes(rotmat, 1, 2)
            return None
        else:
            return coords @ rotmat

    @overload
    def sort(self, sort_by: Union[str, Sequence[int]] = ..., reverse: bool = ..., inplace: Literal[True] = ...) -> None: ...  # type: ignore[misc] # noqa: E501
    @overload
    def sort(self: MT, sort_by: Union[str, Sequence[int]] = ..., reverse: bool = ..., inplace: Literal[False] = ...) -> MT: ...  # noqa: E501
    def sort(self, sort_by='symbol', reverse=False, inplace=True):  # noqa: E301
        """Sort the atoms in this instance and **self.atoms**, performing in inplace update.

        Parameters
        ----------
        sort_by : :class:`str` or :class:`Sequence[int] <collections.abc.Sequence>`
            The property which is to be used for sorting.
            Accepted values: ``"symbol"`` (*i.e.* alphabetical), ``"atnum"``, ``"mass"``,
            ``"radius"`` or ``"connectors"``.
            See the plams.PeriodicTable_ module for more details.
            Alternatively, a user-specified sequence of indices can be provided for sorting.
        reverse : :class:`bool`
            Sort in reversed order.
        inplace : :class:`bool`
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        :class:`FOX.MultiMolecule` or :data:`None`
            If **inplace** is :data:`True`, return a new :class:`MultiMolecule` instance.

        """
        # Create and, potentially, sort a list of indices
        if isinstance(sort_by, str):
            sort_by_array = self._get_atomic_property(prop=sort_by)
            _idx_range = range(self.shape[0])
            idx_range = np.array([i for _, i in sorted(zip(sort_by_array, _idx_range))])
        else:
            idx_range = np.asarray(sort_by)
            assert sort_by.shape[0] == self.shape[1]

        # Reverse or not
        if reverse:
            idx_range.reverse()

        # Inplace update or return a copy
        if inplace:
            mol = self
        else:
            mol = self.copy()

        # Sort this instance
        mol[:] = mol[:, idx_range]

        # Refill **self.atoms**
        symbols = mol.symbol[idx_range]
        atoms_dct = {}
        for i, at in enumerate(symbols):
            try:
                atoms_dct[at].append(i)
            except KeyError:
                atoms_dct[at] = [i]
        mol.atoms = atoms_dct

        # Sort **self.bonds**
        if mol.bonds is not None:
            mol.atom1 = idx_range[mol.atom1]
            mol.atom2 = idx_range[mol.atom2]
            mol.bonds[:, 0:2].sort(axis=1)
            idx = mol.bonds[:, 0:2].argsort(axis=0)[:, 0]
            mol.bonds = mol.bonds[idx]

        # Inplace update or return a copy
        if inplace:
            return None
        else:
            return mol

    @overload
    def residue_argsort(self, concatenate: Literal[True] = ...) -> np.ndarray: ...
    @overload
    def residue_argsort(self, concatenate: Literal[False]) -> List[List[int]]: ...
    def residue_argsort(self, concatenate=True):  # noqa: E301
        """Return the indices that would sort this instance by residue number.

        Residues are defined based on moleculair fragments based on **self.bonds**.

        Parameters
        ----------
        concatenate : :class:`bool`
            If :data:`False`, returned a nested list with atomic indices.
            Each sublist contains the indices of a single residue.

        Returns
        -------
        :class:`np.ndarray[np.int64] <numpy.ndarray>`, shape :math:`(n,)`
            A 1D array of indices that would sort :math:`n` atoms this instance.

        """
        # Define residues
        plams_mol = self.as_Molecule(mol_subset=0)[0]
        frags = separate_mod(plams_mol)
        symbol = self.symbol

        # Sort the residues
        core = []
        ligands = []
        for frag in frags:
            if len(frag) == 1:
                core += frag
            else:
                i = np.array(frag)
                argsort = np.argsort(symbol[i])
                ligands.append(i[argsort].tolist())
        core.sort()
        ligands.sort()

        ret = [core] + ligands
        if concatenate:
            return np.concatenate(ret)
        return ret

    def get_center_of_mass(self, mol_subset: MolSubset = None,
                           atom_subset: AtomSubset = None) -> np.ndarray:
        """Get the center of mass.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, 3)`
            A 2D array with the centres of mass of :math:`m` molecules with :math:`n` atoms.

        """
        coords = self.as_mass_weighted(mol_subset, atom_subset)
        return coords.sum(axis=1) / self.mass.sum()

    def get_bonds_per_atom(self, atom_subset: AtomSubset = None) -> np.ndarray:
        """Get the number of bonds per atom in this instance.

        Parameters
        ----------
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :math:`n` |np.ndarray|_ [|np.int64|_]:
            A 1D array with the number of bonds per atom, for all :math:`n` atoms in this instance.

        """
        j = self._get_atom_subset(atom_subset, as_array=True)
        if self.bonds is None:
            return np.zeros(len(j), dtype=int)
        return np.bincount(self.atom12.ravel(), minlength=self.shape[1])[j]

    """################################## Root Mean Squared ###################################"""

    def _get_time_averaged_prop(self, method: Callable,
                                atom_subset: AtomSubset = None,
                                **kwargs: Any) -> pd.DataFrame:
        r"""A method for constructing time-averaged properties.

        Parameters
        ----------
        method : :class:`Callable[..., ArrayLike] <collections.abc.Callable>`
            A function, method or class used for constructing a specific time-averaged property.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        \**kwargs : :data:`~typing.Any`
            Keyword arguments that will be supplied to **method**.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A DataFrame containing a time-averaged property.

        """
        # Prepare arguments
        loop, at_subset = self._get_loop(atom_subset)

        # Get the time-averaged property
        if loop:
            data = [method(atom_subset=at, **kwargs) for at in at_subset]
        else:
            data = method(atom_subset=at_subset, **kwargs)

        # Construct and return the dataframe
        idx = pd.RangeIndex(0, self.shape[1], name='Abritrary atomic index')
        column_range, data = self._get_rmsf_columns(data, idx, loop=loop, atom_subset=at_subset)
        columns = pd.Index(column_range, name='Atoms')
        return pd.DataFrame(data, index=idx, columns=columns)

    def _get_average_prop(self, method: Callable,
                          atom_subset: AtomSubset = None,
                          **kwargs: Any) -> pd.DataFrame:
        r"""A method for constructing properties averaged over atomic subsets.

        Parameters
        ----------
        Method : :class:`Callable[..., ArrayLike] <collections.abc.Callable>`
            A function, method or class used for constructing a specific atomic subset-averaged
            property.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        \**kwargs : :data:`~typing.Any`
            Keyword arguments that will be supplied to **method**.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A DataFrame containing an atomic subset-averaged property.

        """
        # Prpare arguments
        loop, at_subset = self._get_loop(atom_subset)

        # Calculate and averaged property
        if loop:
            data = np.array([method(atom_subset=at, **kwargs) for at in at_subset]).T
        else:
            data = method(atom_subset=atom_subset, **kwargs).T

        # Construct and return the dataframe
        column_range = self._get_rmsd_columns(loop, atom_subset)
        columns = pd.Index(column_range, name='Atoms')
        return pd.DataFrame(data, columns=columns)

    def init_average_velocity(self, timestep: float = 1.0,
                              rms: bool = False,
                              mol_subset: MolSubset = None,
                              atom_subset: AtomSubset = None) -> pd.DataFrame:
        """Calculate the average atomic velocty.

        The average velocity (in fs/A) is calculated for all atoms in **atom_subset** over the
        course of a trajectory.

        The velocity is averaged over all atoms in a particular atom subset.

        Parameters
        ----------
        timestep : :class:`float`
            The stepsize, in femtoseconds, between subsequent frames.
        rms : :class:`bool`
            Calculate the root-mean squared average velocity instead.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if
            :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A dataframe holding :math:`m-1` velocities averaged over one or more atom subsets.

        """
        kwargs = {'mol_subset': mol_subset, 'timestep': timestep, 'rms': rms}
        df = self._get_average_prop(self.get_average_velocity, atom_subset, **kwargs)
        df.index.name = 'Time / fs'
        return df

    def init_time_averaged_velocity(self, timestep: float = 1.0,
                                    rms: bool = False,
                                    mol_subset: MolSubset = None,
                                    atom_subset: AtomSubset = None) -> pd.DataFrame:
        """Calculate the time-averaged velocty.

        The time-averaged velocity (in fs/A) is calculated for all atoms in **atom_subset** over the
        course of a trajectory.

        Parameters
        ----------
        timestep : :class:`float`
            The stepsize, in femtoseconds, between subsequent frames.
        rms : :class:`bool`
            Calculate the root-mean squared time-averaged velocity instead.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A dataframe holding :math:`m-1` time-averaged velocities.

        """
        kwargs = {'mol_subset': mol_subset, 'timestep': timestep, 'rms': rms}
        return self._get_time_averaged_prop(self.get_time_averaged_velocity, atom_subset, **kwargs)

    def init_rmsd(self, mol_subset: MolSubset = None,
                  atom_subset: AtomSubset = None,
                  reset_origin: bool = True) -> pd.DataFrame:
        """Initialize the RMSD calculation, returning a dataframe.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per  molecule in this instance if :data:`None`.
        reset_origin : :class:`bool`
            Reset the origin of each molecule in this instance by means of
            a partial Procrustes superimposition, translating and rotating the molecules.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A dataframe of RMSDs with one column for every string or list of ints in
            **atom_subset**.
            Keys consist of atomic symbols (*e.g.* ``"Cd"``) if **atom_subset** contains strings,
            otherwise a more generic 'series ' + str(int) scheme is adopted (*e.g.* ``"series 2"``).
            Molecular indices are used as index.

        """
        if reset_origin:
            self.reset_origin()
        kwargs = {'mol_subset': mol_subset}
        df = self._get_average_prop(self.get_rmsd, atom_subset, **kwargs)
        df.index.name = 'XYZ frame number'
        return df

    def init_rmsf(self, mol_subset: MolSubset = None,
                  atom_subset: AtomSubset = None,
                  reset_origin: bool = True) -> pd.DataFrame:
        """Initialize the RMSF calculation, returning a dataframe.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        reset_origin : :class:`bool`
            Reset the origin of each molecule in this instance by means of
            a partial Procrustes superimposition, translating and rotating the molecules.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A dataframe of RMSFs with one column for every string or list of ints in
            **atom_subset**.
            Keys consist of atomic symbols (*e.g.* ``"Cd"``) if **atom_subset** contains strings,
            otherwise a more generic 'series ' + str(int) scheme is adopted (*e.g.* ``"series 2"``).
            Molecular indices are used as indices.

        """
        if reset_origin:
            self.reset_origin()
        kwargs = {'mol_subset': mol_subset}
        return self._get_time_averaged_prop(self.get_rmsf, atom_subset, **kwargs)

    def get_average_velocity(self, timestep: float = 1.0,
                             rms: bool = False,
                             mol_subset: MolSubset = None,
                             atom_subset: AtomSubset = None) -> np.ndarray:
        """Return the mean or root-mean squared velocity.

        Parameters
        ----------
        timestep : :class:`float`
            The stepsize, in femtoseconds, between subsequent frames.
        rms : :class:`bool`
            Calculate the root-mean squared average velocity instead.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if
            :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m-1,)`
            A 1D array holding :math:`m-1` velocities averaged over one or more atom subsets.

        """
        if not rms:
            return self.get_velocity(timestep, mol_subset=mol_subset,
                                     atom_subset=atom_subset).mean(axis=1)
        else:
            v = self.get_velocity(timestep, mol_subset=mol_subset, atom_subset=atom_subset)
            return MultiMolecule(v, self.atoms).get_rmsd(mol_subset)

    def get_time_averaged_velocity(self, timestep: float = 1.0,
                                   rms: bool = False,
                                   mol_subset: MolSubset = None,
                                   atom_subset: AtomSubset = None) -> np.ndarray:
        """Return the mean or root-mean squared velocity (mean = time-averaged).

        Parameters
        ----------
        timestep : :class:`float`
            The stepsize, in femtoseconds, between subsequent frames.
        rms : :class:`bool`
            Calculate the root-mean squared average velocity instead.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n,)`
            A 1D array holding :math:`n` time-averaged velocities.

        """
        if not rms:
            return self.get_velocity(timestep, mol_subset=mol_subset,
                                     atom_subset=atom_subset).mean(axis=0)
        else:
            v = self.get_velocity(timestep, mol_subset=mol_subset, atom_subset=atom_subset)
            return MultiMolecule(v, self.atoms).get_rmsf(mol_subset)

    def get_velocity(self, timestep: float = 1.0,
                     norm: bool = True,
                     mol_subset: MolSubset = None,
                     atom_subset: AtomSubset = None) -> np.ndarray:
        """Return the atomic velocties.

        The velocity (in fs/A) is calculated for all atoms in **atom_subset** over the course of a
        trajectory.

        Parameters
        ----------
        timestep : :class:`float`
            The stepsize, in femtoseconds, between subsequent frames.
        norm : :class:`bool`
            If :data:`True` return the norm of the :math:`x`, :math:`y` and :math:`z`
            velocity components.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, n)` or :math:`(m, n, 3)`
            A 2D or 3D array of atomic velocities, the number of dimensions depending on the
            value of **norm** (:data:`True` = 2D; :data:`False` = 3D).

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Slice the XYZ array and reset the origin
        xyz = self[i, j].reset_origin(inplace=False)

        if norm:
            return np.gradient(np.linalg.norm(xyz, axis=2), timestep, axis=0)
        else:
            return np.gradient(xyz, timestep, axis=0)

    def get_rmsd(self, mol_subset: MolSubset = None,
                 atom_subset: AtomSubset = None) -> np.ndarray:
        """Calculate the root mean square displacement (RMSD).

        The RMSD is calculated with respect to the first molecule in this instance.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A dataframe with the RMSD as a function of the XYZ frame numbers.

        """
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Calculate and return the RMSD per molecule in this instance
        dist = np.linalg.norm(self[i, j, :] - self[0, j, :], axis=2)
        return np.sqrt(np.einsum('ij,ij->i', dist, dist) / dist.shape[1])

    def get_rmsf(self, mol_subset: MolSubset = None,
                 atom_subset: AtomSubset = None) -> np.ndarray:
        """Calculate the root mean square fluctuation (RMSF).

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A dataframe with the RMSF as a function of atomic indices.

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Calculate the RMSF per molecule in this instance
        mean_coords = np.mean(self[i, j, :], axis=0)[None, ...]
        displacement = np.linalg.norm(self[i, j, :] - mean_coords, axis=2)**2
        return np.mean(displacement, axis=0)

    @overload
    @staticmethod
    def _get_rmsd_columns(
        loop: Literal[False], atom_subset: Union[None, str, int, slice, Sequence[int]] = ...
    ) -> Sequence[Hashable]: ...
    @overload  # noqa: E301
    @staticmethod
    def _get_rmsd_columns(
        loop: Literal[True], atom_subset: Union[slice, Sequence[int]]
    ) -> Sequence[Hashable]: ...
    @staticmethod  # noqa: E301
    def _get_rmsd_columns(loop, atom_subset=None):
        """Return the columns for the RMSD dataframe.

        Parameters
        ----------
        loop : :class:`bool`
            If :data:`True`, return a single column name.
            If :data:`False`, return a sequence with multiple column names.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`Sequence[str|int] <collections.abc.Sequence>`
            A sequence with column names for atomic subset-averaged properties.

        """
        if loop:  # Plan A: **atom_subset** is a *list* of *str* or nested *list* of *int*
            if isinstance(atom_subset[0], str):  # Use atomic symbols or general indices as keys
                columns = atom_subset
            else:
                columns = np.arange(len(atom_subset))
        else:  # Plan B: **atom_subset** is something else
            if isinstance(atom_subset, str):  # Use an atomic symbol or a general index as keys
                columns = [atom_subset]
            else:
                columns = [0]

        return columns

    @overload
    def _get_rmsf_columns(
        self,
        rmsf: np.ndarray,
        index: Union[pd.Index, pd.Series, np.ndarray],
        loop: Literal[False],
        atom_subset: Union[None, int, slice, str, Sequence[int]] = ...,
    ) -> Tuple[Sequence[Hashable], np.ndarray]: ...
    @overload  # noqa: E301
    def _get_rmsf_columns(
        self,
        rmsf: np.ndarray,
        index: Union[pd.Index, pd.Series, np.ndarray],
        loop: Literal[False],
        atom_subset: Union[Sequence[str], Sequence[Sequence[int]]],
    ) -> Tuple[Sequence[Hashable], np.ndarray]: ...
    def _get_rmsf_columns(self, rmsf, index, loop, atom_subset=None):  # noqa: E301
        """Return the columns and data for the RMSF dataframe.

        Parameters
        ----------
        rmsf : :class:`np.ndarray[np.float64] <numpy.ndarray>`
            An array with a time-veraged property.
        index : :class:`pd.Index <pandas.Index>`
            The index for the time-averaged property.
        loop : :class:`bool`
            If :data:`True`, return a single column name.
            If :data:`False`, return a sequence with multiple column names.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`Sequence[str|int]` and :class:`np.ndarray[np.float64] <numpy.ndarray>`
            A sequence with column names for time-averaged properties
            and an array with the time-averaged data.

        """
        if loop:  # Plan A: **atom_subset** is a *list* of *str* or nested *list* of *int*
            if isinstance(atom_subset[0], str):  # Use atomic symbols or general indices as keys
                columns = atom_subset
            else:
                columns = np.arange(len(atom_subset))

            # Create and fill a padded array
            data = np.full((len(rmsf), index.shape[0]), np.nan)
            k = 0
            for i, j in enumerate(rmsf):
                data[i, k:k+len(j)] = j
                k += len(j)
            data = data.T

        else:  # Plan B: **atom_subset** is something else
            if isinstance(atom_subset, str):  # Use an atomic symbol or a general index as keys
                columns = [atom_subset]
            else:
                columns = [0]

            # Create and fill a padded array
            data = np.full((index.shape[0]), np.nan)
            idx = self._get_atom_subset(atom_subset)
            data[idx] = rmsf

        return columns, data

    # TODO remove this method in favor of MultiMolecule._get_at_iterable()
    @overload
    def _get_loop(
        self, atom_subset: Union[None, range, slice, int, Sequence[int]]
    ) -> Tuple[Literal[False], Union[Sequence[int], slice]]: ...
    @overload  # noqa: E301
    def _get_loop(
        self, atom_subset: Union[Sequence[str], Sequence[Sequence[int]]]
    ) -> Tuple[Literal[True], Sequence[Sequence[int]]]: ...
    def _get_loop(self, atom_subset):  # noqa: E301
        """Figure out if the supplied subset warrants a for loop or not.

        Parameters
        ----------
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`bool` and :class:`np.ndarray[np.int64] <numpy.ndarray>`
            A boolean and (nested) iterable consisting of integers.

        """
        if atom_subset is None:
            return False, slice(None)
        elif isinstance(atom_subset, (range, slice)):
            return False, atom_subset
        elif isinstance(atom_subset, str):
            return False, self.atoms[atom_subset]
        elif isinstance(atom_subset, int):
            return False, [atom_subset]
        elif isinstance(atom_subset[0], (int, np.integer)):
            return False, atom_subset
        elif isinstance(atom_subset[0], str):
            return True, [self.atoms[i] for i in atom_subset]
        elif isinstance(atom_subset[0][0], (int, np.integer)):
            return True, atom_subset

        err = "'{}' of type '{}' is an invalid argument for 'atom_subset'"
        raise TypeError(err.format(str(atom_subset), atom_subset.__class__.__name__))

    """#############################  Determining shell structures  ##########################"""

    def init_shell_search(
        self,
        mol_subset: MolSubset = None,
        atom_subset: AtomSubset = None,
        rdf_cutoff: float = 0.5
    ) -> NoReturn:
        """Calculate and return properties which can help determining shell structures.

        Warning
        -------
        Depercated.

        """  # noqa: E501
        cls = type(self)
        raise DeprecationWarning(f"`{cls.__name__}.init_shell_search` is deprecated")

    @staticmethod
    def get_at_idx(
        rmsf: pd.DataFrame,
        idx_series: pd.Series,
        dist_dict: Dict[str, List[float]],
    ) -> NoReturn:
        """Create subsets of atomic indices.

        Warning
        -------
        Depercated.

        """
        raise DeprecationWarning("`MultiMolecule.get_at_idx` is deprecated")

    """#############################  Radial Distribution Functions  ##########################"""

    def init_rdf(
        self,
        mol_subset: MolSubset = None,
        atom_subset: AtomSubset = None,
        *,
        dr: float = 0.05,
        r_max: float = 12.0,
        periodic: None | Sequence[Literal["x", "y", "z"]] | Sequence[Literal[0, 1, 2]] = None,
        atom_pairs: None | Iterable[tuple[str, str]] = None,
    ) -> pd.DataFrame:
        """Initialize the calculation of radial distribution functions (RDFs).

        RDFs are calculated for all possible atom-pairs in **atom_subset** and returned as a
        dataframe.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        dr : :class:`float`
            The integration step-size in Ångström, *i.e.* the distance between
            concentric spheres.
        r_max : :class:`float`
            The maximum to be evaluated interatomic distance in Ångström.
        periodic : :class:`str`, optional
            If specified, correct for the systems periodicity if
            :attr:`self.lattice is not None <MultiMolecule.lattice>`.
            Accepts ``"x"``, ``"y"`` and/or ``"z"``.
        atom_pairs : :class:`Iterable[tuple[str, str]] <collections.abc.Iterable>`
            An explicit list of atom-pairs for the to-be calculated distances.
            Note that **atom_pairs** and **atom_subset** are mutually exclusive.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A dataframe of radial distribution functions, averaged over all conformations in
            **xyz_array**.
            Keys are of the form: at_symbol1 + ' ' + at_symbol2 (*e.g.* ``"Cd Cd"``).
            Radii are used as index.

        """
        if atom_subset is not None and atom_pairs is not None:
            raise TypeError("`atom_subset` and `atom_pairs` are mutually exclusive")
        elif atom_pairs is not None:
            pair_dict = _parse_atom_pairs(self, atom_pairs)
        elif atom_subset is not None:
            pair_dict = self.get_pair_dict(atom_subset, r=2)
        else:
            # If **atom_subset** is None: extract atomic symbols from they keys of **self.atoms**
            pair_dict = self.get_pair_dict(sorted(self.atoms, key=str), r=2)

        # Construct an empty dataframe with appropiate dimensions, indices and keys
        df = get_rdf_df(pair_dict, dr, r_max)

        # Define the subset
        m_subset = self._get_mol_subset(mol_subset)
        m_self = self[m_subset]

        # Parse the lattice and periodicty settings
        if periodic is not None:
            periodic_ar = parse_periodic(periodic)
            if self.lattice is None:
                raise TypeError("cannot perform periodic calculations if the "
                                "molecules `lattice` is None")
            lattice_ar = self.lattice if self.lattice.ndim == 2 else self.lattice[m_subset]
            volume = lattice_to_volume(lattice_ar)
        else:
            volume = None
            lattice_ar = _GetNone()
            periodic_ar = np.arange(3, dtype=np.int64)

        # Fill the dataframe with RDF's, averaged over all conformations in this instance
        n_mol = len(m_self)
        for key, (i, j) in pair_dict.items():
            shape = n_mol, len(i), len(j)
            iterator = slice_iter(shape, m_self.dtype.itemsize)
            for slc in iterator:
                dist_mat = m_self.get_dist_mat(
                    mol_subset=slc, atom_subset=(i, j),
                    lattice=lattice_ar[slc], periodicity=periodic_ar,
                )
                df[key] += get_rdf(dist_mat, dr=dr, r_max=r_max, volume=volume)
        df /= n_mol
        return df

    def get_dist_mat(
        self,
        mol_subset: MolSubset = None,
        atom_subset: Tuple[AtomSubset, AtomSubset] = (None, None),
        lattice: None | np.ndarray[Any, np.dtype[np.float64]] = None,
        periodicity: Iterable[Literal[0, 1, 2]] = range(3),
    ) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Create and return a distance matrix for all molecules and atoms in this instance.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        lattice : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(3, 3)` or :math:`(m, 3, 3)`, optional
            If not :data:`None`, use the specified lattice vectors for
            correcting periodic effects.
        periodicty : :class:`str`
            The axes along which the system's periodicity extends; accepts
            ``"x"``, ``"y"`` and/or ``"z"``.
            Only relevant if ``lattice is not None``.

        Returns
        -------
        :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, n, k)`
            A 3D distance matrix of :math:`m` molecules, created out of two sets of :math:`n`
            and :math:`k` atoms.

        """  # noqa: E501
        # Define array slices
        m_subset = self._get_mol_subset(mol_subset)

        # Slice the XYZ array
        A = self[m_subset, self._get_atom_subset(atom_subset[0])]
        B = self[m_subset, self._get_atom_subset(atom_subset[1])]

        # Create, fill and return the distance matrix
        if A.ndim == 2:
            return cdist(A, B)[None, ...]

        if lattice is None:
            shape = A.shape[0], A.shape[1], B.shape[1]
            ret = np.empty(shape, dtype=self.dtype)
            for k, (a, b) in enumerate(zip(A, B)):
                ret[k] = cdist(a, b)
            return ret

        ret = np.abs(A[..., None, :] - B[..., None, :, :])
        lat_norm = np.linalg.norm(lattice, axis=-1)
        if lat_norm.ndim == 1:
            iterator = ((i, lat_norm[i]) for i in periodicity)
            for i, ar1 in iterator:
                ret[..., i][ret[..., i] > (ar1 / 2)] -= ar1
        elif lat_norm.ndim == 2:
            iterator = ((i, lat_norm[:, i]) for i in periodicity)
            for i, _ar2 in iterator:
                ar2 = np.full(ret.shape[:-1], _ar2[..., None, None])
                condition = ret[..., i] > (ar2 / 2)
                ret[..., i][condition] -= ar2[condition]
        else:
            raise ValueError
        return np.linalg.norm(ret, axis=-1)

    def get_pair_dict(self, atom_subset: Union[Sequence[AtomSubset],
                                               Mapping[Hashable, Sequence[AtomSubset]]],
                      r: int = 2) -> Dict[str, Tuple[np.ndarray, ...]]:
        """Take a subset of atoms and return a dictionary.

        Parameters
        ----------
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        r : :class:`int`
            The length of the to-be returned subsets.

        """
        if isinstance(atom_subset, abc.Mapping):
            key_iter = (str(i) for i in atom_subset.keys())
            value_iter = (self._get_atom_subset(i) for i in atom_subset.values())
        else:
            key_iter = ((j if isinstance(j, abc.Hashable) else i) for i, j in enumerate(atom_subset))  # noqa
            value_iter = (self._get_atom_subset(i) for i in atom_subset)

        key_gen = combinations_with_replacement(key_iter, r)
        value_gen = combinations_with_replacement(value_iter, r)

        ret = {}
        iterator = (zip(permutations(k), permutations(v)) for k, v in zip(key_gen, value_gen))
        for kv in iterator:
            for k, v in kv:
                if not (k in ret or k[::-1] in ret):
                    ret[k] = v
        return {' '.join(str(i) for i in k): v for k, v in ret.items()}

    """####################################  Power spectrum  ###################################"""

    def init_power_spectrum(
        self,
        mol_subset: MolSubset = None,
        atom_subset: AtomSubset = None,
        freq_max: int = 4000,
        timestep: float = 1,
    ) -> pd.DataFrame:
        """Calculate and return the power spectrum associated with this instance.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        freq_max : :class:`int`
            The maximum to be returned wavenumber (cm**-1).
        timestep : :class:`float`
            The stepsize, in femtoseconds, between subsequent frames.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A DataFrame containing the power spectrum for each set of atoms in
            **atom_subset**.

        """
        def slice_iter(iterable: Iterable[tuple[_T, Sized]]) -> Iterator[tuple[_T, slice]]:
            i = j = 0
            for name, sized in iterable:
                j += len(sized)
                yield name, slice(i, j)
                i += len(sized)

        # Construct the velocity autocorrelation function
        vacf = self.get_vacf(mol_subset, atom_subset, timestep)

        # Create the to-be returned DataFrame
        freq_max = int(freq_max) + 1
        idx = pd.RangeIndex(0, freq_max, name='Wavenumber / cm**-1')
        df = pd.DataFrame(index=idx)

        # Construct power spectra intensities
        n = int(1 / (constants.c * 1e-13))
        power_complex = fft(vacf, n, axis=0) / len(vacf)
        power_abs = np.abs(power_complex)

        iterator = slice_iter(self._get_at_iterable(atom_subset))
        for at, slc in iterator:
            power_slice = power_abs[:, slc]
            df[at] = np.einsum('ij,ij->i', power_slice, power_slice)[:freq_max]
        return df

    def get_vacf(
        self,
        mol_subset: MolSubset = None,
        atom_subset: AtomSubset = None,
        timestep: float = 1,
    ) -> np.ndarray:
        """Calculate and return the velocity autocorrelation function (VACF).

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        timestep : :class:`float`
            The stepsize, in femtoseconds, between subsequent frames.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A DataFrame containing the power spectrum for each set of atoms in
            **atom_subset**.

        """
        from scipy.signal import fftconvolve

        # Get atomic velocities
        v = self.get_velocity(  # A / s
            timestep * 1e-15, mol_subset=mol_subset, atom_subset=atom_subset
        )

        # Construct the velocity autocorrelation function
        vacf = fftconvolve(v, v[::-1], axes=0)[len(v)-1:]
        dv = v - v.mean(axis=0)
        return vacf / np.einsum('ij,ij->j', dv, dv)

    def _get_at_iterable(self, atom_subset: AtomSubset) -> Iterable[Tuple[Hashable, Any]]:
        """Return an iterable that returns 2-tuples upon iteration.

        The **atom_subset** argument is evaluated and converted into an iterable.
        Upon iteration, this iterable yields 2-tuples consisting of:
        1. A hashable intended for the column names of DataFrames.
        2. An object for slicing arrays.

        If **atom_subset** consists of one or more string (*i.e.*) atoms, then those while be used
        as hashable. An enumeration scheme will be employed otherwise.

        Examples
        --------
        .. code:: python

            >>> import numpy as np
            >>> from FOX import (MultiMolecule, get_example_xyz)

            >>> np.set_printoptions(threshold=10)
            >>> mol = MultiMolecule.from_xyz(get_example_xyz())

            >>> atom_subset = ['C', 'H', 'O']
            >>> atom_iter = mol._get_at_iterable(atom_subset)
            >>> for at, idx in atom_iter:
            >>>     print(at, np.array(idx))
            C [123 127 131 ... 215 219 223]
            H [124 128 132 ... 216 220 224]
            O [125 126 129 ... 222 225 226]

            >>> atom_subset = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
            >>> atom_iter = mol._get_at_iterable(atom_subset)
            >>> for at, idx in atom_iter:
            >>>     print(at, np.array(idx))
            0 [1 2 3]
            1 [4 5 6]
            2 [7 8 9]


        Parameters
        ----------
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`Iterable[tuple[int|str, Sequence[int]]] <collections.abc.Iterable>`
            A boolean and (nested) iterable consisting of integers.

        Raises
        ------
        TypeError
            Raised if **atom_subset** is of an invalid type.

        """
        if atom_subset is None:
            return self.atoms.items()
        elif isinstance(atom_subset, (range, slice)):
            return enumerate((atom_subset,))
        elif isinstance(atom_subset, str):
            return [(atom_subset, self.atoms[atom_subset])]
        elif isinstance(atom_subset, int):
            return [(False, (atom_subset,))]
        elif len(atom_subset) == 0:
            return []
        elif isinstance(atom_subset[0], (int, np.integer)):
            return enumerate(atom_subset)
        elif isinstance(atom_subset[0], str):
            return [(at, self.atoms[at]) for at in atom_subset]
        elif isinstance(atom_subset[0][0], (int, np.integer)):
            return enumerate(atom_subset)

        err = "'{}' of type '{}' is an invalid argument for 'atom_subset'"
        raise TypeError(err.format(str(atom_subset), atom_subset.__class__.__name__))

    """############################  Angular Distribution Functions  ##########################"""

    def init_adf(
        self,
        mol_subset: MolSubset = None,
        atom_subset: AtomSubset = None,
        *,
        r_max: Union[float, str] = 8.0,
        weight: Callable[[np.ndarray], np.ndarray] = neg_exp,
        periodic: None | Sequence[Literal["x", "y", "z"]] | Sequence[Literal[0, 1, 2]] = None,
        atom_pairs: None | Iterable[tuple[str, str]] = None,
    ) -> pd.DataFrame:
        r"""Initialize the calculation of distance-weighted angular distribution functions (ADFs).

        ADFs are calculated for all possible atom-pairs in **atom_subset** and returned as a
        dataframe.

        .. _DASK: https://dask.org/

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        r_max : :class:`float`
            The maximum inter-atomic distance (in Angstrom) for which angles are constructed.
            The distance cuttoff can be disabled by settings this value to ``np.inf``, ``"np.inf"``
            or ``"inf"``.
        weight : :class:`Callable[[np.ndarray], np.ndarray] <collections.abc.Callable>`, optional
            A callable for creating a weighting factor from inter-atomic distances.
            The callable should take an array as input and return an array.
            Given an angle :math:`\phi_{ijk}`, to the distance :math:`r_{ijk}` is defined
            as :math:`max[r_{ij}, r_{jk}]`.
            Set to :data:`None` to disable distance weighting.
        periodic : :class:`str`, optional
            If specified, correct for the systems periodicity if
            :attr:`self.lattice is not None <MultiMolecule.lattice>`.
            Accepts ``"x"``, ``"y"`` and/or ``"z"``.
        atom_pairs : :class:`Iterable[tuple[str, str, str]] <collections.abc.Iterable>`
            An explicit list of atom-triples for the to-be calculated angles.
            Note that **atom_pairs** and **atom_subset** are mutually exclusive.

        Returns
        -------
        :class:`pd.DataFrame <pandas.DataFrame>`
            A dataframe of angular distribution functions, averaged over all conformations in
            this instance.

        Note
        ----
        Disabling the distance cuttoff is strongly recommended (*i.e.* it is faster) for large
        values of **r_max**. As a rough guideline, ``r_max="inf"`` is roughly as fast as
        ``r_max=15.0`` (though this is, of course, system dependant).

        Note
        ----
        The ADF construction will be conducted in parralel if the DASK_ package is installed.
        DASK can be installed, via anaconda, with the following command:
        ``conda install -n FOX -y -c conda-forge dask``.

        """
        # Identify the maximum to-be considered radius
        r_max_ = 0 if float(r_max) == np.inf else float(r_max)
        n = int((1 + r_max_ / 4)**3)

        # Identify atom and molecule subsets
        m_subset = self._get_mol_subset(mol_subset)
        if atom_subset is not None and atom_pairs is not None:
            raise TypeError("`atom_subset` and `atom_pairs` are mutually exclusive")
        elif atom_pairs is not None:
            if not isinstance(atom_pairs, abc.Collection):
                atom_pairs = list(atom_pairs)
            at_subset = self._get_atom_subset(list(chain.from_iterable(atom_pairs)), as_array=True)
        else:
            at_subset = self._get_atom_subset(atom_subset, as_array=True)

        # Slice this MultiMolecule instance based on **atom_subset** and **mol_subset**
        del_atom = np.ones(self.shape[1], dtype=bool)
        del_atom[at_subset] = False
        mol = self.delete_atoms(del_atom)[m_subset]

        if atom_pairs is not None:
            at_pairs = _parse_atom_pairs(self, atom_pairs)
        elif atom_subset is not None:
            at_pairs = mol.get_pair_dict(atom_subset or sorted(mol.atoms, key=str), r=3)
        else:
            at_pairs = mol.get_pair_dict(sorted(mol.atoms, key=str), r=3)

        for k, v in at_pairs.items():
            v_new = []
            for at in v:
                bool_ar = np.zeros(mol.shape[1], dtype=bool)
                bool_ar[mol.atoms[at] if isinstance(at, str) else at] = True
                v_new.append(bool_ar)
            at_pairs[k] = v_new

        # import pdb; pdb.set_trace()

        # Periodic calculations
        if periodic is not None:
            periodic_ar = parse_periodic(periodic)

            # Validate the parameters
            lattice = self.lattice
            if lattice is None:
                raise TypeError("cannot perform periodic calculations if the "
                                "molecules `lattice` is None")
            else:
                lattice = cast("np.ndarray[Any, np.dtype[np.float64]]", lattice[m_subset])

            # Set the vector-length of all absent axes to `inf`
            slc = [i for i in range(3) if i not in periodic_ar]
            lattice[..., slc, :] = np.inf

            lattice_iter = repeat(lattice) if lattice.ndim == 2 else iter(lattice)
            periodic_iter = repeat(periodic_ar)
        else:
            lattice_iter = repeat(None)
            periodic_iter = repeat(range(3))

        # Construct the angular distribution function
        # Perform the task in parallel (with dask) if possible
        if DASK_EX is None and r_max_:
            func = dask.delayed(_adf_inner_cdktree)
            jobs = [func(m, n, r_max_, at_pairs.values(), l, p, weight) for m, l, p in
                    zip(mol, lattice_iter, periodic_iter)]
            results = dask.compute(*jobs)
        elif DASK_EX is None and not r_max_:
            func = dask.delayed(_adf_inner)
            jobs = [func(m, at_pairs.values(), l, p, weight) for m, l, p in
                    zip(mol, lattice_iter, periodic_iter)]
            results = dask.compute(*jobs)
        elif DASK_EX is not None and r_max_:
            func = _adf_inner_cdktree
            results = [func(m, n, r_max_, at_pairs.values(), l, p, weight) for m, l, p in
                       zip(mol, lattice_iter, periodic_iter)]
        elif DASK_EX is not None and not r_max_:
            func = _adf_inner
            results = [func(m, at_pairs.values(), l, p, weight) for m, l, p in
                       zip(mol, lattice_iter, periodic_iter)]

        df = get_adf_df(at_pairs)
        df.loc[:, :] = np.array(results).mean(axis=0).T
        return df

    @overload
    def _get_atom_subset(self, atom_subset: AtomSubset, as_array: Literal[False] = ...) -> Union[slice, np.ndarray[Any, np.dtype[np.intp]]]: ...  # noqa: E501
    @overload
    def _get_atom_subset(self, atom_subset: AtomSubset, as_array: Literal[True]) -> np.ndarray[Any, np.dtype[np.intp]]: ...  # noqa: E501
    def _get_atom_subset(self, atom_subset, as_array=False):  # noqa: E301
        """Sanitize the **_get_atom_subset** argument.

        Accepts the following objects:

            * :data:`None`
            * ``range`` or ``slice`` instances
            * Integers
            * Strings (*i.e.* atom types; see the **MultiMolecule.atoms** attribute)
            * Sequence of integers (*e.g.* lists, tuples or arrays)
            * Sequence of strings
            * Nested sequence of integers
            * Boolean sequences

        Notes
        -----
        Supports object suitable for both fancy and non-fancy array indexing.

        Parameters
        ----------
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        as_array : :class:`bool`
            Ensure the subset is returned as an array of integers.

        Returns
        -------
        :class:`Sequence[int] <collections.abc.Sequence>` or :class:`np.ndarray[np.bool_|np.intp] <numpy.ndarray>`
            An object suitable for array slicing.

        Raises
        ------
        TypeError
            Raised if an object unsuitable for array slicing is provided.

        """  # noqa: E501
        if as_array:
            if atom_subset is None:
                return np.arange(0, self.shape[-2])
            elif isinstance(atom_subset, (range, slice)):
                return np.arange(atom_subset.start, atom_subset.stop, atom_subset.step)
        else:
            if atom_subset is None:
                return slice(None)
            elif isinstance(atom_subset, slice):
                return atom_subset
            elif isinstance(atom_subset, range):
                return slice(atom_subset.start, atom_subset.stop, atom_subset.step)

        ret = np.array(atom_subset, ndmin=1, copy=False).ravel()
        try:
            i = ret[0]
        except IndexError:  # Empty sequence
            return np.empty((0,), dtype=np.intp)

        if isinstance(i, np.str_):
            return np.concatenate([self._atoms_get(j) for j in ret]).astype(np.intp, copy=False)
        elif isinstance(i, np.integer):
            return ret.astype(np.intp, copy=False)
        elif isinstance(i, np.bool_):
            return ret if not as_array else np.arange(len(ret), dtype=np.intp)[ret]

        # A Collection or Iterator; try harder
        ret2 = np.array(list(chain.from_iterable(ret))).ravel()
        j = ret2[0]
        if isinstance(j, np.str_):
            return np.concatenate([self._atoms_get(j) for j in ret2]).astype(np.intp, copy=False)
        elif isinstance(j, np.integer):
            return ret2.astype(np.intp, copy=False)
        elif isinstance(j, np.bool_):
            return ret2 if not as_array else np.arange(len(ret2), dtype=np.intp)[ret]

        raise TypeError(f"'atom_subset' is of invalid type: '{atom_subset.__class__.__name__}'")

    def _get_mol_subset(self, mol_subset: MolSubset) -> slice:
        """Sanitize the **mol_subset** argument.

        Accepts the following objects:

            * :data:`None`
            * ``range`` or ``slice`` instances
            * Integers

        Notes
        -----
        Objects suitable for fancy array indexing are *not* supported.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.

        Returns
        -------
        :class:`slice` or :class:`int`:
            An object suitable for array slicing.

        Raises
        ------
        TypeError
            Raised if **mol_subset** is of invalid type.

        """
        if mol_subset is None:
            return slice(None)
        elif isinstance(mol_subset, slice):
            return mol_subset
        elif hasattr(mol_subset, '__index__'):
            try:
                i = mol_subset.__index__()
                if mol_subset >= 0:
                    return slice(i, i + 1)
                else:
                    return slice(i, None)
            except TypeError as ex:
                err = "The 'mol_subset' parameter cannot be used as scalar inder"
                raise ValueError(err).with_traceback(ex.__traceback__)

        raise TypeError(f"'mol_subset' is of invalid type: '{mol_subset.__class__.__name__}'")

    """#################################  Type conversion  ####################################"""

    def _mol_to_file(self, filename: Union[str, PathLike],
                     outputformat: Optional[str] = None,
                     mol_subset: Optional[MolSubset] = 0) -> None:
        """Create files using the :class:`plams.Molecule.write <scm.plams.mol.molecule.Molecule.write>` method.

        Parameters
        ----------
        filename : :term:`python:path-like`
            The path+filename (including extension) of the to be created file.
        outputformat : :class:`str`
            The outputformat.
            Accepated values are ``"mol"``, ``"mol2"``, ``"pdb"`` or ``"xyz"``.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.

        """  # noqa: E501
        m_subset = self._get_mol_subset(mol_subset)
        mol_range = range(m_subset.start or 0, m_subset.stop or len(self), m_subset.step or 1)
        outputformat = outputformat or str(filename).rsplit('.', 1)[-1]
        plams_mol = self.as_Molecule(mol_subset=0)[0]

        if len(mol_range) != 1 or (mol_range.stop - mol_range.start) // mol_range.step != 1:
            name_list = str(filename).rsplit('.', 1)
            name_list.insert(-1, '.{:d}.')
            name = ''.join(name_list)
        else:
            name = str(filename)

        from_array = plams_mol.from_array
        write = plams_mol.write
        for i, j in enumerate(mol_range, 1):
            from_array(self[j])
            write(name.format(i), outputformat=outputformat)

    def as_pdb(self, filename: Union[str, PathLike],
               mol_subset: Optional[MolSubset] = 0) -> None:
        """Convert a *MultiMolecule* object into one or more Protein DataBank files (.pdb).

        Utilizes the :class:`plams.Molecule.write <scm.plams.mol.molecule.Molecule.write>` method.

        Parameters
        ----------
        filename : :term:`python:path-like object`
            The path+filename (including extension) of the to be created file.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.

        """
        self._mol_to_file(filename, 'pdb', mol_subset)

    def as_mol2(self, filename: Union[str, PathLike],
                mol_subset: Optional[MolSubset] = 0) -> None:
        """Convert a :class:`FOX.MultiMolecule` object into one or more .mol2 files.

        Utilizes the :class:`plams.Molecule.write <scm.plams.mol.molecule.Molecule.write>` method.

        Parameters
        ----------
        filename : :term:`python:path-like object`
            The path+filename (including extension) of the to be created file.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.

        """
        self._mol_to_file(filename, 'mol2', mol_subset)

    def as_mol(self, filename: Union[str, PathLike],
               mol_subset: Optional[MolSubset] = 0) -> None:
        """Convert a *MultiMolecule* object into one or more .mol files.

        Utilizes the :class:`plams.Molecule.write <scm.plams.mol.molecule.Molecule.write>` method.

        Parameters
        ----------
        filename : :term:`python:path-like object`
            The path+filename (including extension) of the to be created file.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.

        """  # noqa: E501
        self._mol_to_file(filename, 'mol', mol_subset)

    def as_xyz(self, filename: Union[str, PathLike],
               mol_subset: Optional[MolSubset] = None) -> None:
        """Create an .xyz file out of this instance.

        Comments will be constructed by iteration through ``MultiMolecule.properties["comments"]``
        if the following two conditions are fulfilled:

        * The ``"comments"`` key is actually present in :attr:`MultiMolecule.properties`.
        * ``MultiMolecule.properties["comments"]`` is an iterable.

        Parameters
        ----------
        filename : :term:`python:path-like object`
            The path+filename (including extension) of the to be created file.
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.

        """
        # Define constants and variables
        m_subset = self[self._get_mol_subset(mol_subset)].astype(str)
        at = self.symbol[:, None]
        header = '{:d}\n'.format(len(at))
        kwargs = {
            'fmt': ['%-10.10s', '%-15s', '%-15s', '%-15s'], 'delimiter': 5 * ' ', 'comments': ''
        }

        # Alter variables depending on the presence or absence of self.properties.comments
        if 'comments' in self.properties and isinstance(self.properties.comments, abc.Iterable):
            header += '{}'
            iterator = islice(zip_longest(self.properties.comments, m_subset), len(m_subset))
        else:
            header += 'Frame {:d}'
            iterator = enumerate(m_subset, 1)

        # Create the .xyz file
        with open(filename, 'wb') as file:
            for i, xyz in iterator:
                np.savetxt(file, np.hstack((at, xyz)), header=header.format(i), **kwargs)

    @overload
    def as_mass_weighted(self: MT, mol_subset: MolSubset = ..., atom_subset: AtomSubset = ..., inplace: Literal[False] = ...) -> MT: ...  # type: ignore[misc] # noqa: E501
    @overload
    def as_mass_weighted(self, mol_subset: MolSubset = ..., atom_subset: AtomSubset = ..., inplace: Literal[True] = ...) -> None: ...  # noqa: E501
    def as_mass_weighted(self, mol_subset=None, atom_subset=None, inplace=False):  # noqa: E301
        """Transform the Cartesian of this instance into mass-weighted Cartesian coordinates.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        inplace : :class:`bool`
            Instead of returning the new coordinates, perform an inplace update of this instance.

        Returns
        -------
        :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(m, n, 3)`, optional
            if **inplace** = :data:`False` return a new :class:`.MultiMolecule` instance with the
            mass-weighted Cartesian coordinates of :math:`m` molecules with :math:`n` atoms.

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Create an array of mass-weighted Cartesian coordinates
        if inplace:
            self[i, j, :] *= self.mass[None, j, None]
            return None
        else:
            return self[i, j, :] * self.mass[None, j, None]

    def from_mass_weighted(self, mol_subset: MolSubset = None,
                           atom_subset: AtomSubset = None) -> None:
        """Transform this instance from mass-weighted Cartesian into Cartesian coordinates.

        Performs an inplace update of this instance.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Update this instance
        self[i, j, :] /= self.mass[None, j, None]

    def as_Molecule(self, mol_subset: MolSubset = None,
                    atom_subset: AtomSubset = None) -> List[Molecule]:
        """Convert this instance into a *list* of *plams.Molecule*.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.

        Returns
        -------
        :class:`list[plams.Molecule] <list>`
            A list of :math:`m` PLAMS molecules constructed from this instance.

        """
        m_subset = self._get_mol_subset(mol_subset)
        at_subset = self._get_atom_subset(atom_subset, as_array=True)
        at_subset.sort()
        at_symbols = self.symbol

        # Construct a template molecule and fill it with atoms
        mol_template = Molecule()
        mol_template.properties = self.properties.copy()
        add_atom = mol_template.add_atom
        for i in at_subset:
            atom = Atom(symbol=at_symbols[i])
            add_atom(atom)

        # Fill the template molecule with bonds
        if self.bonds.any():
            bond_idx = np.ones(self.shape[-2], dtype=int)
            bond_idx[at_subset] += np.arange(len(at_subset))
            bond_idx = bond_idx.tolist()

            add_bond = mol_template.add_bond
            for i, j, order in self.bonds:
                if i in at_subset and j in at_subset:
                    at1 = mol_template[bond_idx[i]]
                    at2 = mol_template[bond_idx[j]]
                    add_bond(Bond(atom1=at1, atom2=at2, order=order/10.0))

        if self.lattice is None:
            lattice_iter = ([] for _ in self[m_subset])
        elif self.lattice.ndim == 2:
            lattice_iter = (self.lattice.tolist() for _ in self[m_subset])
        elif self.lattice.ndim == 3:
            lattice_iter = iter(self.lattice.tolist())
        else:
            raise ValueError

        # Create copies of the template molecule; update their cartesian coordinates
        ret: List[Molecule] = []
        ret_append = ret.append
        for i, (lat, xyz) in enumerate(zip(lattice_iter, self[m_subset])):
            mol = mol_template.copy()
            mol.from_array(xyz[at_subset])
            mol.properties.frame = i
            mol.lattice = lat
            ret_append(mol)

        return ret

    @classmethod
    def from_Molecule(cls: Type[MT], mol_list: Union[Molecule, Sequence[Molecule]],
                      subset: None | Container[str] = frozenset({'atoms'})) -> MT:
        """Construct a :class:`.MultiMolecule` instance from one or more PLAMS molecules.

        Parameters
        ----------
        mol_list : :class:`plams.Molecule <scm.plams.mol.molecule.Molecule>` or :class:`Sequence[plams.Molecule] <collections.abc.Sequence>`
            A PLAMS molecule or list of PLAMS molecules.
        subset : :class:`Container[str] <collections.abc.Container>`, optional
            Transfer a subset of *plams.Molecule* attributes to this instance.
            If :data:`None`, transfer all attributes.
            Accepts one or more of the following values as strings:
            ``"properties"``, ``"atoms"``, ``"lattice"`` and/or ``"bonds"``.

        Returns
        -------
        :class:`FOX.MultiMolecule`:
            A molecule constructed from **mol_list**.

        """  # noqa: E501
        if isinstance(mol_list, Molecule):
            plams_mol = mol_list
            mol_list = (mol_list,)
        else:
            plams_mol = mol_list[0]
        subset = subset if subset is not None else {'atoms', 'bonds', 'properties', 'lattice'}

        # Convert coordinates
        n_mol = len(mol_list)
        n_atom = len(plams_mol)
        iterator = chain.from_iterable(at.coords for mol in mol_list for at in mol)
        coords = np.fromiter(iterator, dtype=float, count=n_mol*n_atom*3)
        coords.shape = n_mol, n_atom, 3

        kwargs: dict = {}

        # Convert atoms
        if 'atoms' in subset:
            iterator_ = ((i, at.symbol) for i, at in enumerate(plams_mol.atoms))
            kwargs['atoms'] = group_by_values(iterator_)

        # Convert properties
        if 'properties' in subset:
            kwargs['properties'] = plams_mol.properties.copy()

        # Convert bonds
        if 'bonds' in subset:
            plams_mol.set_atoms_id(start=0)
            kwargs['bonds'] = np.array([(bond.atom1.id, bond.atom2.id, bond.order * 10) for
                                        bond in plams_mol.bonds], dtype=int)
            plams_mol.unset_atoms_id()

        # Convert lattice
        if 'lattice' in subset:
            lat = np.array([m.lattice for m in mol_list], dtype=np.float64)
            kwargs['lattice'] = lat if lat.size != 0 else None
        return cls(coords, **kwargs)

    def as_ase(self, mol_subset: MolSubset = None,
               atom_subset: AtomSubset = None, **kwargs: Any) -> List[Atoms]:
        r"""Convert this instance into a list of ASE :class:`~ase.Atoms`.

        Parameters
        ----------
        mol_subset : :class:`slice`, optional
            Perform the calculation on a subset of molecules in this instance, as
            determined by their moleculair index.
            Include all :math:`m` molecules in this instance if :data:`None`.
        atom_subset : :class:`Sequence[str] <collections.abc.Sequence>`, optional
            Perform the calculation on a subset of atoms in this instance, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in this instance if :data:`None`.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for :class:`ase.Atoms`.

        Returns
        -------
        :class:`list[ase.Atoms] <list>`
            A list of ASE Atoms constructed from this instance.

        """
        if ASE_EX is not None:
            raise ASE_EX

        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        symbols = self.symbol[j]
        positions_iter = self[i, j]

        if "cell" in kwargs:
            lattice_iter = repeat(kwargs.pop("cell"))
        else:
            if self.lattice is None:
                lattice_iter = repeat(None)
            elif self.lattice.ndim == 2:
                lattice_iter = repeat(self.lattice)
            elif self.lattice.ndim == 3:
                lattice_iter = iter(self.lattice[i])
            else:
                raise ValueError

        return [Atoms(symbols=symbols, positions=p, cell=lat, **kwargs) for lat, p in
                zip(lattice_iter, positions_iter)]

    @classmethod
    def from_ase(cls: Type[MT], mol_list: Union[Atoms, Sequence[Atoms]]) -> MT:
        """Construct a :class:`.MultiMolecule` instance from one or more ASE Atoms.

        Parameters
        ----------
        mol_list : :class:`ase.Atoms` or :class:`Sequence[ase.Atoms] <collections.abc.Sequence>`
            An ASE Atoms instance or a list thereof.

        Returns
        -------
        :class:`FOX.MultiMolecule`:
            A molecule constructed from **mol_list**.

        """  # noqa: E501
        if ASE_EX is not None:
            raise ASE_EX

        if isinstance(mol_list, Atoms):
            mol_list = cast("List[Atoms]", [mol_list])
        elif len(mol_list) == 0:
            raise ValueError("`mol_list` should contain at least one molecule")

        coords = [m.positions for m in mol_list]
        _atoms = group_by_values(enumerate(mol_list[0].numbers))
        atoms = {PeriodicTable.get_symbol(k): v for k, v in _atoms.items()}

        lattice = np.array([m.cell for m in mol_list], dtype=np.float64)
        if not lattice.any():
            lattice = None
        return cls(coords, atoms=atoms, lattice=lattice)

    @classmethod
    def from_xyz(cls: Type[MT], filename: Union[str, bytes, PathLike],
                 bonds: Optional[np.ndarray] = None,
                 properties: Optional[dict] = None,
                 read_comment: bool = False) -> MT:
        """Construct a :class:`.MultiMolecule` instance from a (multi) .xyz file.

        Comment lines extracted from the .xyz file are stored, as array, under
        ``MultiMolecule.properties["comments"]``.

        Parameters
        ----------
        filename : :term:`python:path-like object`
            The path+filename of an .xyz file.
        bonds : :class:`np.ndarray[np.int64] <numpy.ndarray>`, shape :math:`(k, 3)`
            An optional 2D array with indices of the atoms defining all :math:`k` bonds
            (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
            Stored in the **MultieMolecule.bonds** attribute.
        properties : :class:`dict`, optional
            A Settings object (subclass of dictionary) intended for storing
            miscellaneous user-defined (meta-)data. Is devoid of keys by default.
            Stored in the **MultiMolecule.properties** attribute.
        read_comments : :class:`bool`
            If :data:`True`, extract all comment lines from the passed .xyz file and
            store them under :attr:`properties.comments<MultiMolecule.properties>`.

        Returns
        -------
        :class:`FOX.MultiMolecule`
            A molecule constructed from **filename**.

        """
        if read_comment:
            coords, atoms, comments = read_multi_xyz(filename, return_comment=True)
        else:
            coords, atoms = read_multi_xyz(filename, return_comment=False)

        ret = cls(coords, atoms, bonds, properties)
        ret.properties['filename'] = filename
        if read_comment:
            ret.properties['comments'] = comments
        return ret

    @classmethod
    def from_kf(cls: Type[MT], filename: Union[str, 'PathLike[str]'],
                bonds: Optional[np.ndarray] = None,
                properties: Optional[dict] = None) -> MT:
        """Construct a :class:`.MultiMolecule` instance from a KF binary file.

        Parameters
        ----------
        filename : :term:`python:path-like object`
            The path+filename of an KF binary file.
        bonds : :class:`np.ndarray[np.int64] <numpy.ndarray>`, shape :math:`(k, 3)`
            An optional 2D array with indices of the atoms defining all :math:`k` bonds
            (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
            Stored in the **MultieMolecule.bonds** attribute.
        properties : :class:`dict`
            A Settings object (subclass of dictionary) intended for storing
            miscellaneous user-defined (meta-)data. Is devoid of keys by default.
            Stored in the **MultiMolecule.properties** attribute.

        Returns
        -------
        :class:`FOX.MultiMolecule`
            A molecule constructed from **filename**.

        """
        ret = cls(*read_kf(filename), bonds, properties)
        ret.properties['filename'] = filename
        return ret
