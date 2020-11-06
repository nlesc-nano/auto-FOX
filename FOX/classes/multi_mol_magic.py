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
    Dict, Optional, List, Any, Callable, Union, Mapping, Iterable,
    NoReturn, TypeVar, Type, Generic, cast
)

import numpy as np

from scm.plams import PeriodicTable, PTError, Settings  # type: ignore
from assertionlib.ndrepr import NDRepr
from nanoutils import ArrayLike, Literal

__all__: List[str] = []

T = TypeVar('T')
MT = TypeVar('MT', bound='_MultiMolecule')

IdxDict = Dict[str, List[int]]
IdxMapping = Mapping[str, List[int]]

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


class _MolLoc(Generic[MT]):
    """A getter and setter for atom-type-based slicing.

    Get, set and del operations are performed using the list(s) of atomic indices associated
    with the provided atomic symbol(s).
    Accepts either one or more atomic symbols.

    Examples
    --------
    .. code:: python

        >>> mol = MultiMolecule(...)
        >>> mol.atoms['Cd'] = [0, 1, 2, 3, 4, 5]
        >>> mol.atoms['Se'] = [6, 7, 8, 9, 10, 11]
        >>> mol.atoms['O'] = [12, 13, 14]

        >>> (mol.loc['Cd'] == mol[mol.atoms['Cd']]).all()
        True

        >>> idx = mol.atoms['Cd'] + mol.atoms['Se'] + mol.atoms['O']
        >>> (mol.loc['Cd', 'Se', 'O'] == mol[idx]).all()
        True

        >>> mol.loc['Cd'] = 1
        >>> print((mol.loc['Cd'] == 1).all())
        True

        >>> del mol.loc['Cd']
        ValueError: cannot delete array elements


    Parameters
    ----------
    mol : :class:`~FOX.MultiMolecule`
        A MultiMolecule instance; see :attr:`_MolLoc.mol`.

    Attributes
    ----------
    mol : :class:`~FOX.MultiMolecule`
        A MultiMolecule instance.
    atoms_view : :class:`~collections.abc.Mapping`
        A read-only view of :attr:`_MolLoc.mol.atoms<FOX.MultiMolecule.atoms>`.

    """

    __slots__ = ('_mol', '_atoms')

    @property
    def mol(self) -> MT:
        return self._mol

    @property
    def atoms_view(self) -> IdxMapping:
        try:
            return self._atoms
        except AttributeError:
            self._atoms: IdxMapping = MappingProxyType(self.mol.atoms)
            return self._atoms

    def __init__(self, mol: MT) -> None:
        self._mol = mol

    def _type_error(self, obj: object) -> TypeError:
        """Return a :exc:`TypeError`."""
        cls_name = self.mol.__class__.__name__
        name = obj.__class__.__name__
        return TypeError(f"{cls_name}.loc() expected one or more strings; observed type: {name!r}")

    def __getitem__(self, key: Union[str, Iterable[str]]) -> MT:
        """Get items from :attr:`_MolLoc.mol`."""
        idx = self._parse_key(key)
        return self.mol[..., idx, :]

    def __setitem__(self, key: Union[str, Iterable[str]], value: ArrayLike) -> None:
        """Set items in :attr:`_MolLoc.mol`."""
        idx = self._parse_key(key)
        self.mol[..., idx, :] = value

    def __delitem__(self, key: Union[str, Iterable[str]]) -> NoReturn:
        """Delete items from :attr:`_MolLoc.mol`; this raises a :exc:`ValueError`."""
        idx = self._parse_key(key)
        del self.mol[..., idx, :]  # This will raise a ValueError
        raise

    def _parse_key(self, key: Union[str, Iterable[str]]) -> np.ndarray:
        """Return the atomic indices of **key** are all atoms in **key**.

        Parameter
        ---------
        key : :class:`str` or :class:`Iterable[str]<collections.abc.Iterable>`
            An atom type or an iterable consisting of atom types.

        Returns
        -------
        :class:`list` [:class:`int`]
            A (flattened) list of atomic indices.

        """
        if isinstance(key, str):
            return self.atoms_view[key]

        try:
            key_iterator = iter(key)
        except TypeError as ex:  # **key** is neither a string nor an iterable of strings
            raise self._type_error(key) from ex

        # Gather all indices and flatten them
        idx: List[int] = []
        atoms = self.atoms_view
        for k in key_iterator:
            idx += atoms[k]
        return idx

    def __reduce__(self) -> NoReturn:
        raise TypeError(f'cannot pickle {self.__class__.__name__!r} object')

    def __hash__(self) -> int:
        """Implement :code:`hash(self)`."""
        return id(self.mol)

    def __eq__(self, value: Any) -> bool:
        """Implement :code:`self == value`."""
        try:
            return type(self) is type(value) and self.mol is value.mol
        except AttributeError:
            return False


class _MultiMolecule(np.ndarray):
    """Private superclass of :class:`.MultiMolecule`.

    Handles all magic methods and @property decorated methods.

    """

    def __new__(cls: Type[MT], coords: np.ndarray,
                atoms: Optional[IdxDict] = None,
                bonds: Optional[np.ndarray] = None,
                properties: Optional[Mapping] = None) -> MT:
        """Create and return a new object."""
        obj = np.array(coords, dtype=float, ndmin=3, copy=False).view(cls)

        # Set attributes
        obj.atoms = cast(IdxDict, atoms)
        obj.bonds = cast(np.ndarray, bonds)
        obj.properties = cast(Settings, properties)
        obj._ndrepr = NDRepr()
        return obj

    def __array_finalize__(self, obj: _MultiMolecule) -> None:
        """Finalize the array creation."""
        if obj is None:
            return

        self.atoms = getattr(obj, 'atoms', None)
        self.bonds = getattr(obj, 'bonds', None)
        self.properties = getattr(obj, 'properties', None)
        self._ndrepr = getattr(obj, '_ndrepr', None)

    """#####################  Properties for managing instance attributes  ######################"""

    @property
    def loc(self: MT) -> _MolLoc[MT]:
        return _MolLoc(self)
    loc.__doc__ = _MolLoc.__doc__

    @property
    def atoms(self) -> IdxDict:
        return self._atoms

    @atoms.setter
    def atoms(self, value: Optional[Mapping[str, List[int]]]) -> None:
        self._atoms = {} if value is None else dict(value)

    @property
    def bonds(self) -> np.ndarray:
        return self._bonds

    @bonds.setter
    def bonds(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            bonds = np.zeros((0, 3), dtype=int)
        else:
            bonds = np.array(value, dtype=int, ndmin=2, copy=False)

        # Set bond orders to 1 (i.e. 10 / 10) if the order is not specified
        if bonds.shape[1] == 2:
            order = np.full(len(bonds), fill_value=10, dtype=int)[..., None]
            self._bonds = np.hstack([bonds, order])
        else:
            self._bonds = bonds

    @property
    def properties(self) -> Settings:
        return self._properties

    @properties.setter
    def properties(self, value: Optional[Mapping]) -> None:
        self._properties = Settings() if value is None else Settings(value)

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
        prop : str
            The name of the to be returned property.
            Accepted values: ``"symbol"``, ``"atnum"``, ``"mass"``, ``"radius"``
            or ``"connectors"``.
            See the |PeriodicTable|_ class of PLAMS for more details.

        Returns
        -------
        :math:`n` |np.array|_ [|np.float64|_, |str|_ or |np.int64|_]:
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

    def copy(self: MT, order: str = 'C', deep: bool = True) -> MT:
        """Create a copy of this instance.

        .. _np.ndarray.copy: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.copy.html  # noqa

        Parameters
        ----------
        order : str
            Controls the memory layout of the copy.
            See np.ndarray.copy_ for details.

        copy_attr : bool
            Whether or not the attributes of this instance should be returned as copies or views.

        Returns
        -------
        |FOX.MultiMolecule|_:
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
