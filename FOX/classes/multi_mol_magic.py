"""A Module for setting up the magic methods and properties of the :class:`.MultiMolecule` class.

Index
-----
.. currentmodule:: FOX.classes.multi_mol_magic
.. autosummary::
    _MultiMolecule

API
---
.. autoclass:: _MultiMolecule
    :members:
    :noindex:

"""

from __future__ import annotations

import copy as pycopy
import textwrap
import itertools
import warnings
from types import MappingProxyType
from typing import (
    Dict, Optional, List, Any, Callable, Union, Mapping, Iterable, Tuple,
    NoReturn, TypeVar, Type, Generic, cast, TYPE_CHECKING, Sequence, NamedTuple
)

import numpy as np

from scm.plams import PeriodicTable, PTError, Settings  # type: ignore
from assertionlib.ndrepr import NDRepr
from nanoutils import ArrayLike, Literal

if TYPE_CHECKING:
    import numpy.typing as npt

    _ArLikeInt = Union[int, Sequence[int], npt._SupportsArray[np.dtype[np.integer[Any]]]]
    IdxDict = MappingProxyType[str, np.ndarray[Any, np.dtype[np.intp]]]
    IdxMapping = Union[
        Mapping[str, _ArLikeInt],
        Iterable[Tuple[str, _ArLikeInt]],
    ]

    AliasDict = MappingProxyType[str, AliasTuple]  # noqa: F821
    AliasMapping = Mapping[str, Tuple[str, Union[slice, ellipsis, _ArLikeInt]]]  # noqa: F821

__all__: List[str] = ["_MultiMolecule", "AliasTuple"]


class AliasTuple(NamedTuple):
    """A 2-tuple used for :attr:`FOX.MultiMolecule.atoms` values."""

    alias: str
    slice: Union[slice, ellipsis, np.ndarray[Any, np.dtype[np.intp]]]  # noqa: F821


T = TypeVar('T')
MT = TypeVar('MT', bound='_MultiMolecule')

_PROP_MAPPING: Mapping[str, Callable[[str], Union[int, float, str]]] = MappingProxyType({
    'symbol': lambda x: x,
    'radius': PeriodicTable.get_radius,
    'atnum': PeriodicTable.get_atomic_number,
    'mass': PeriodicTable.get_mass,
    'connectors': PeriodicTable.get_connectors
})

_NONE_DICT: Mapping[str, Union[str, int, float]] = MappingProxyType({
    'symbol': '', 'radius': -1, 'atnum': -1, 'mass': np.nan, 'connectors': -1
})


def _to_int_array(ar: _ArLikeInt) -> np.ndarray[Any, np.dtype[np.intp]]:
    _ret = np.array(ar, ndmin=1, copy=False)
    if _ret.dtype == bool:
        raise TypeError("Expected an integer array")

    ret = _ret.astype(np.intp, copy=False, casting="same_kind")
    if ret.ndim != 1:
        raise ValueError("Expected a <= 1D array")
    if ret.base is not None:
        return ret.copy()
    return ret


def _parse_lattice(a: npt.ArrayLike) -> np.ndarray[Any, np.dtype[np.float64]]:
    ar = np.asarray(a).astype(np.float64, casting="same_kind", copy=False)
    if ar.ndim not in {2, 3}:
        raise ValueError("`lattice` expected a 2D or 3D array; "
                         f"observed dimensionality: {ar.ndim!r}")
    elif ar.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid `lattice` shape: {ar.shape!r}")
    return ar


class _MolLoc(Generic[MT]):
    """A getter and setter for atom-type-based slicing.

    Get, set and del operations are performed using the list(s) of atomic indices associated
    with the provided atomic symbol(s).
    Accepts either one or more atomic symbols.

    Examples
    --------
    .. code:: python

        >>> mol = MultiMolecule(...)
        >>> mol.atoms = {
        ...     'Cd': [0, 1, 2, 3, 4, 5],
        ...     'Se': [6, 7, 8, 9, 10, 11],
        ...     'O': [12, 13, 14],
        ... }

        >>> (mol.loc['Cd'] == mol[mol.atoms['Cd']]).all()
        True

        >>> idx = []
        >>> for atom in ["Cd", "Se", "O"]:
        ...     idx += mol.atoms[atom].tolist()
        >>> (mol.loc['Cd', 'Se', 'O'] == mol[idx]).all()
        True

        >>> mol.loc['Cd'] = 1
        >>> print((mol.loc['Cd'] == 1).all())
        True

        >>> del mol.loc['Cd']
        ValueError: cannot delete array elements


    Parameters
    ----------
    mol : :class:`FOX.MultiMolecule`
        A MultiMolecule instance; see :attr:`_MolLoc.mol`.

    Attributes
    ----------
    mol : :class:`FOX.MultiMolecule`
        A MultiMolecule instance.
    atoms_view : :class:`~collections.abc.Mapping`
        A read-only view of :attr:`_MolLoc.mol.atoms<FOX.MultiMolecule.atoms>`.

    """

    __slots__ = ('_mol',)

    @property
    def mol(self) -> MT:
        return self._mol

    @property
    def atoms(self) -> IdxDict:
        return self.mol.atoms

    def __init__(self, mol: MT) -> None:
        self._mol = mol

    def _type_error(self, obj: object) -> TypeError:
        """Return a :exc:`TypeError`."""
        cls_name = self.mol.__class__.__name__
        name = obj.__class__.__name__
        return TypeError(f"{cls_name}.loc() expected one or more strings; observed type: {name!r}")

    def __getitem__(self, key: Union[str, Tuple[str, ...]]) -> MT:
        """Get items from :attr:`_MolLoc.mol`."""
        idx = self._parse_key(key)
        return self.mol[..., idx, :]

    def __setitem__(self, key: Union[str, Tuple[str, ...]], value: ArrayLike) -> None:
        """Set items in :attr:`_MolLoc.mol`."""
        idx = self._parse_key(key)
        self.mol[..., idx, :] = value

    def __delitem__(self, key: Union[str, Tuple[str, ...]]) -> NoReturn:
        """Delete items from :attr:`_MolLoc.mol`; this raises a :exc:`ValueError`."""
        # This will raise a ValueError
        idx = self._parse_key(key)
        del self.mol[..., idx, :]  # type: ignore[attr-defined]
        raise

    def _parse_key(self, key: Union[str, Sequence[str]]) -> np.ndarray[Any, np.dtype[np.intp]]:
        """Return the atomic indices of **key** are all atoms in **key**.

        Parameter
        ---------
        key : :class:`str` or :class:`Iterable[str] <collections.abc.Iterable>`
            An atom type or an iterable consisting of atom types.

        Returns
        -------
        :class:`list[int] <list>`
            A (flattened) list of atomic indices.

        """
        if isinstance(key, str):
            return self.atoms[key].copy()

        try:
            key_iterator = iter(key)
        except TypeError as ex:  # **key** is neither a string nor an iterable of strings
            raise self._type_error(key) from ex

        # Gather all indices and flatten them
        lst = [self.mol._atoms_get(k) for k in key_iterator]
        if len(lst):
            return np.concatenate(lst).astype(np.intp, copy=False)
        else:
            return np.array([], dtype=np.intp)

    def __reduce__(self) -> NoReturn:
        raise TypeError(f'cannot pickle {self.__class__.__name__!r} object')

    def __hash__(self) -> int:
        """Implement :code:`hash(self)`."""
        return id(self.mol)

    def __eq__(self, value: object) -> bool:
        """Implement :code:`self == value`."""
        if type(self) is not type(value):
            return NotImplemented
        return self.mol is value.mol


class _MultiMolecule(np.ndarray):
    """Private superclass of :class:`.MultiMolecule`.

    Handles all magic methods and @property decorated methods.

    """

    def __new__(
        cls: Type[MT],
        coords: npt.ArrayLike,
        atoms: Optional[IdxMapping] = None,
        bonds: Optional[np.ndarray] = None,
        properties: Optional[Mapping] = None,
        atoms_alias: Optional[Mapping[str, slice]] = None,
        lattice: None | npt.ArrayLike = None,
    ) -> MT:
        """Create and return a new object."""
        obj = np.array(coords, dtype=np.float64, ndmin=3, copy=False).view(cls)

        # Set attributes
        obj.atoms = cast("IdxDict", atoms)
        obj.bonds = cast(np.ndarray, bonds)
        obj.properties = cast("Settings", properties)
        obj.atoms_alias = cast("AliasDict", atoms_alias)
        obj.lattice = cast("Optional[np.ndarray[Any, np.dtype[np.float64]]]", lattice)
        obj._ndrepr = NDRepr()
        return obj

    def __array_finalize__(self, obj: _MultiMolecule) -> None:
        """Finalize the array creation."""
        if obj is None:
            return

        self.atoms = getattr(obj, 'atoms', None)
        self.bonds = getattr(obj, 'bonds', None)
        self.properties = getattr(obj, 'properties', None)
        self.atoms_alias = getattr(obj, 'atoms_alias', None)
        self.lattice = getattr(obj, 'lattice', None)
        self._ndrepr = getattr(obj, '_ndrepr', None)

    """#####################  Properties for managing instance attributes  ######################"""

    @property
    def loc(self: MT) -> _MolLoc[MT]:
        return _MolLoc(self)
    loc.__doc__ = _MolLoc.__doc__

    @property
    def atoms(self) -> IdxDict:
        return MappingProxyType(self._atoms)

    @atoms.setter
    def atoms(self, value: None | IdxMapping) -> None:
        if value is None:
            self._atoms: Dict[str, np.ndarray[Any, np.dtype[np.intp]]] = {}
            return None

        dct = {k: _to_int_array(v) for k, v in dict(value).items()}
        if len(dct):
            ar_tot = np.concatenate(list(dct.values()))
            if len(ar_tot) != len(np.unique(ar_tot)):
                raise ValueError("Expected non-interseting arrays")

        self._atoms = dct
        return None

    @property
    def atoms_alias(self) -> AliasDict:
        return MappingProxyType(self._atoms_alias)

    @atoms_alias.setter
    def atoms_alias(self, value: None | AliasMapping) -> None:
        if value is None:
            self._atoms_alias = {}
            return None

        dct = {}
        for k, (alias, slc) in value.items():
            if alias not in self.atoms:
                raise KeyError(alias)
            elif k in self.atoms:
                raise KeyError(k)
            elif isinstance(slc, slice) or slc is Ellipsis:
                dct[k] = AliasTuple(alias, slc)
            else:
                dct[k] = AliasTuple(alias, _to_int_array(slc))
            _ = self.atoms[alias][dct[k][1]]
        self._atoms_alias = dct
        return None

    @property
    def bonds(self) -> np.ndarray[Any, np.dtype[np.intp]]:
        return self._bonds

    @bonds.setter
    def bonds(self, value: Optional[np.ndarray[Any, np.dtype[np.integer[Any]]]]) -> None:
        if value is None:
            bonds = np.zeros((0, 3), dtype=np.intp)
        else:
            _bonds = np.array(value, ndmin=2, copy=False)
            bonds = _bonds.astype(np.intp, casting="same_kind", copy=False)

        # Set bond orders to 1 (i.e. 10 / 10) if the order is not specified
        if bonds.shape[1] == 2:
            order = np.full(len(bonds), fill_value=10, dtype=np.intp)[..., None]
            self._bonds = np.hstack([bonds, order])
        else:
            self._bonds = bonds

    @property
    def properties(self) -> Settings:
        return self._properties

    @properties.setter
    def properties(self, value: Optional[Mapping[Any, Any]]) -> None:
        self._properties = Settings() if value is None else Settings(value)

    @property
    def lattice(self) -> None | np.ndarray[Any, np.dtype[np.float64]]:
        return self._lattice

    @lattice.setter
    def lattice(self, value: None | npt.ArrayLike) -> None:
        if value is None:
            self._lattice = None
        else:
            self._lattice = _parse_lattice(value)

    """###############################  PLAMS-based properties  ################################"""

    @property
    def atom12(self) -> np.ndarray:
        """Get or set the indices of the atoms for all bonds in :attr:`.MultiMolecule.bonds` as 2D array."""  # noqa
        return self._bonds[:, 0:2]

    @atom12.setter
    def atom12(self, value: np.ndarray) -> None:
        self._bonds[:, 0:2] = value

    @property
    def atom1(self) -> _MultiMolecule:
        """Get or set the indices of the first atoms in all bonds of :attr:`.MultiMolecule.bonds` as 1D array."""  # noqa
        return self._bonds[:, 0]

    @atom1.setter
    def atom1(self, value: np.ndarray) -> None:
        self._bonds[:, 0] = value

    @property
    def atom2(self) -> np.ndarray:
        """Get or set the indices of the second atoms in all bonds of :attr:`.MultiMolecule.bonds` as 1D array."""  # noqa
        return self._bonds[:, 1]

    @atom2.setter
    def atom2(self, value: np.ndarray) -> None:
        self._bonds[:, 1] = value

    @property
    def order(self) -> np.ndarray:
        """Get or set the bond orders for all bonds in :attr:`.MultiMolecule.bonds` as 1D array."""
        return self._bonds[:, 2] / 10.0

    @order.setter
    def order(self, value: np.ndarray) -> None:
        self._bonds[:, 2] = value * 10

    @property
    def x(self: MT) -> MT:
        """Get or set the x coordinates for all atoms in instance as 2D array."""
        return self[:, :, 0]

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self[:, :, 0] = value

    @property
    def y(self: MT) -> MT:
        """Get or set the y coordinates for all atoms in this instance as 2D array."""
        return self[:, :, 1]

    @y.setter
    def y(self, value: np.ndarray) -> None:
        self[:, :, 1] = value

    @property
    def z(self: MT) -> MT:
        """Get or set the z coordinates for all atoms in this instance as 2D array."""
        return self[:, :, 2]

    @z.setter
    def z(self, value: np.ndarray) -> None:
        self[:, :, 2] = value

    @property
    def symbol(self) -> np.ndarray:
        """Get the atomic symbols of all atoms in :attr:`.MultiMolecule.atoms` as 1D array."""
        return self._get_atomic_property('symbol')

    @property
    def atnum(self) -> np.ndarray:
        """Get the atomic numbers of all atoms in :attr:`.MultiMolecule.atoms` as 1D array."""
        return self._get_atomic_property('atnum')

    @property
    def mass(self) -> np.ndarray:
        """Get the atomic masses of all atoms in :attr:`.MultiMolecule.atoms` as 1D array."""
        return self._get_atomic_property('mass')

    @property
    def radius(self) -> np.ndarray:
        """Get the atomic radii of all atoms in :attr:`.MultiMolecule.atoms` as 1d array."""
        return self._get_atomic_property('radius')

    @property
    def connectors(self) -> np.ndarray:
        """Get the atomic connectors of all atoms in :attr:`.MultiMolecule.atoms` as 1D array."""
        return self._get_atomic_property('connectors')

    def _get_atomic_property(
        self, prop: Literal['symbol', 'radius', 'atnum', 'mass', 'connectors']
    ) -> np.ndarray:
        """Create a flattened array with atomic properties.

        Take **self.atoms** and return an (concatenated) array of a specific property associated
        with an atom type. Values are sorted by their indices.

        Parameters
        ----------
        prop : :class:`str`
            The name of the to be returned property.
            Accepted values: ``"symbol"``, ``"atnum"``, ``"mass"``, ``"radius"``
            or ``"connectors"``.
            See the |PeriodicTable|_ class of PLAMS for more details.

        Returns
        -------
        :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n,)`
            A 1D array with the user-specified properties of :math:`n` atoms.

        """
        # Create a concatenated lists of the keys and values in **self.atoms**
        prop_list: list = []
        for at, i in self.atoms.items():
            try:
                at_prop = _PROP_MAPPING[prop](at)
            except PTError:  # A custom atom is encountered
                at_prop = _NONE_DICT[prop]
                warnings.warn(f"KeyWarning: No '{prop}' available for '{at}', "
                              f"defaulting to '{at_prop}'")
            prop_list += [at_prop] * len(i)

        # Sort and return
        idx_gen = itertools.chain.from_iterable(self.atoms.values())
        return np.array([prop for _, prop in sorted(zip(idx_gen, prop_list))])

    """##################################  Magic methods  #################################### """

    def _atoms_get(self, key: str) -> np.ndarray[Any, np.dtype[np.intp]]:
        if key in self.atoms:
            return self.atoms[key]
        else:
            k, slc = self.atoms_alias[key]
            return self.atoms[k][slc]

    def copy(self: MT, order: str = 'C', *, deep: bool = True) -> MT:
        """Create a copy of this instance.

        Parameters
        ----------
        order : :class:`str`
            Controls the memory layout of the copy.
            See :meth:`ndarray.copy <numpy.ndarray.copy>` for details.
        copy_attr : :class:`bool`
            Whether or not the attributes of this instance should be returned as copies or views.

        Returns
        -------
        :class:`FOX.MultiMolecule`
            A copy of this instance.

        """
        ret = super().copy(order)
        if not deep:
            return ret

        # Copy attributes
        copy_func = cast(Callable[[T], T], pycopy.deepcopy if deep else pycopy.copy)
        iterator = copy_func(vars(self)).items()
        for key, value in iterator:
            setattr(ret, key, value)
        return ret

    def __copy__(self: MT) -> MT:
        """Create copy of this instance."""
        return self.copy(order='K', deep=False)

    def __deepcopy__(self: MT, memo: Optional[Dict[int, Any]] = None) -> MT:
        """Create a deep copy of this instance."""
        return self.copy(order='K', deep=True)

    def __str__(self) -> str:
        """Return a human-readable string constructed from this instance."""
        ret = super().__str__()
        ret_indent = textwrap.indent(ret, 4 * ' ')
        return f'{self.__class__.__name__}(\n{ret_indent}\n)'

    def __repr__(self) -> str:
        """Return the canonical string representation of this instance."""
        return f'{self.__class__.__name__}(..., shape={self.shape}, dtype={repr(self.dtype.name)})'

    def __reduce__(self: MT) -> Tuple[MT, Tuple[Any, Any, Any, Any, Any, Any]]:
        """Helper function for :mod:`pickle`."""
        cls = type(self)
        args = (
            self.view(np.ndarray),
            self._atoms,
            self._bonds,
            self._properties,
            self._atoms_alias,
            self._lattice,
        )
        return cls, args
