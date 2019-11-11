"""
FOX.classes.multi_mol_magic
===========================

A Module for setting up the magic methods and properties of the :class:`.MultiMolecule` class.

Index
-----
.. currentmodule:: FOX.classes.multi_mol_magic
.. autosummary::
    _MultiMolecule

API
---
.. autoclass:: _MultiMolecule
    :members:
    :private-members:
    :special-members:

"""

from __future__ import annotations

import copy as pycopy
import textwrap
import itertools
import warnings
from types import MappingProxyType
from typing import Dict, Optional, List, Any, Callable, Union, Mapping

import numpy as np

from scm.plams import PeriodicTable, Atom, PTError, Settings
from assertionlib.ndrepr import NDRepr

__all__: List[str] = []

_PROP_MAPPING: Mapping[str, Callable[[Atom], Any]] = MappingProxyType({
    'symbol': lambda x: x,
    'radius': PeriodicTable.get_radius,
    'atnum': PeriodicTable.get_atomic_number,
    'mass': PeriodicTable.get_mass,
    'connectors': PeriodicTable.get_connectors
})

_NONE_DICT: Mapping[str, Union[str, int, float]] = MappingProxyType({
    'symbol': '', 'radius': -1, 'atnum': -1, 'mass': np.nan, 'connectors': -1
})


class _MultiMolecule(np.ndarray):
    """Private superclass of :class:`.MultiMolecule`.

    Handles all magic methods and @property decorated methods.
    """

    def __new__(cls, coords: np.ndarray,
                atoms: Optional[Dict[str, List[int]]] = None,
                bonds: Optional[np.ndarray] = None,
                properties: Optional[Dict[str, Any]] = None) -> _MultiMolecule:
        """Create and return a new object."""
        obj = np.array(coords, dtype=float, ndmin=3, copy=False).view(cls)

        # Set attributes
        obj.atoms = atoms
        obj.bonds = bonds
        obj.properties = properties
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
    def atoms(self) -> Dict[str, List[int]]:
        return self._atoms

    @atoms.setter
    def atoms(self, value: Optional[Mapping]) -> None:
        self._atoms = {} if value is None else dict(value)

    @property
    def bonds(self) -> np.ndarray:
        return self._bonds

    @bonds.setter
    def bonds(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            self._bonds = np.zeros((0, 3), dtype=int)
        else:
            self._bonds = np.array(value, dtype=int, ndmin=2, copy=False)

    @property
    def properties(self) -> Settings:
        return self._properties

    @properties.setter
    def properties(self, value: Optional[Mapping]) -> None:
        self._properties = Settings() if value is None else Settings(value)

    """###############################  PLAMS-based properties  ################################"""

    @property
    def atom12(self) -> _MultiMolecule:
        """Get or set the indices of the atoms for all bonds in
        :attr:`.MultiMolecule.bonds` as 2D array."""
        return self.bonds[:, 0:2]

    @atom12.setter
    def atom12(self, value: np.ndarray) -> None:
        self.bonds[:, 0:2] = value

    @property
    def atom1(self) -> _MultiMolecule:
        """Get or set the indices of the first atoms in all bonds of
        :attr:`.MultiMolecule.bonds` as 1D array."""
        return self.bonds[:, 0]

    @atom1.setter
    def atom1(self, value: np.ndarray) -> None:
        self.bonds[:, 0] = value

    @property
    def atom2(self) -> np.ndarray:
        """Get or set the indices of the second atoms in all bonds of
        :attr:`.MultiMolecule.bonds` as 1D array."""
        return self.bonds[:, 1]

    @atom2.setter
    def atom2(self, value: np.ndarray) -> None:
        self.bonds[:, 1] = value

    @property
    def order(self) -> np.ndarray:
        """Get or set the bond orders for all bonds in :attr:`.MultiMolecule.bonds` as 1D array."""
        return self.bonds[:, 2] / 10.0

    @order.setter
    def order(self, value: np.ndarray) -> None:
        self.bonds[:, 2] = value * 10

    @property
    def x(self) -> _MultiMolecule:
        """Get or set the x coordinates for all atoms in instance as 2D array."""
        return self[:, :, 0]

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self[:, :, 0] = value

    @property
    def y(self) -> _MultiMolecule:
        """Get or set the y coordinates for all atoms in this instance as 2D array."""
        return self[:, :, 1]

    @y.setter
    def y(self, value: _MultiMolecule) -> None:
        self[:, :, 1] = value

    @property
    def z(self) -> _MultiMolecule:
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

    def _get_atomic_property(self, prop: str = 'symbol') -> np.ndarray:
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

    def copy(self, order: str = 'C', deep: bool = True) -> _MultiMolecule:
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
        copy_func = pycopy.deepcopy if deep else pycopy.copy()
        iterator = copy_func(vars(self)).items()
        for key, value in iterator:
            setattr(ret, key, value)
        return ret

    def __copy__(self) -> _MultiMolecule:
        """Create copy of this instance."""
        return self.copy(order='K', deep=False)

    def __deepcopy__(self, memo: None) -> _MultiMolecule:
        """Create a deep copy of this instance."""
        return self.copy(order='K', deep=True)

    def __str__(self) -> str:
        """Return a human-readable string constructed from this instance."""
        def _str(k: str, v: Any) -> str:
            key = k.strip('_')
            str_list = self._ndrepr.repr(v).split('\n')
            joiner = '\n' + (3 + len(key)) * ' '
            return f'{k} = ' + joiner.join(i for i in str_list)

        ret = f'{self._ndrepr.repr(self)},\n\n'
        ret += ',\n\n'.join(_str(k, v) for k, v in vars(self).items())
        ret_indent = textwrap.indent(ret, '    ')
        return f'{self.__class__.__name__}(\n{ret_indent}\n)'

    def __repr__(self) -> str:
        """Return the canonical string representation of this instance."""
        return f'{self.__class__.__name__}(..., shape={self.shape}, dtype={repr(self.dtype.name)})'
