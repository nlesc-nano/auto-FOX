""" A Module for setting up the magic methods of the MultiMolecule class. """

__all__ = []

import itertools

import numpy as np

from scm.plams import PeriodicTable
from scm.plams.core.settings import Settings


class _MultiMolecule(np.ndarray):
    """ A class for handling the magic methods and
    @property decorated methods of *MultiMolecule*.
    """
    def __new__(cls, coords, atoms=None, bonds=None, properties=None):
        assert isinstance(coords, np.ndarray)
        obj = np.asarray(coords).view(cls)
        atoms, bonds, properties = _MultiMolecule._sanitize_new(atoms, bonds, properties)

        # Set attributes
        obj.atoms = atoms
        obj.bonds = bonds
        obj.properties = properties
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.atoms = getattr(obj, 'atoms', None)
        self.bonds = getattr(obj, 'bonds', None)
        self.properties = getattr(obj, 'properties', None)

    @staticmethod
    def _sanitize_new(atoms, bonds, properties):
        """ A function for sanitizing the arguments of __new__(). """
        # Sanitize **bonds**
        if bonds is not None:
            assert isinstance(bonds, np.ndarray)

        # Sanitize **atoms**
        if atoms is None:
            atoms = {}
        else:
            assert isinstance(atoms, dict)

        # Sanitize **properties**
        if properties is None:
            properties = Settings({'atom': None, 'bond': None, 'Molecule': None})
        else:
            assert isinstance(properties, dict)

        return atoms, bonds, properties

    """ ##############################  plams-based properties  ############################### """

    @property
    def atom12(self):
        """ Return the indices of the atoms for all bonds in **self.bonds** as 2d array"""
        return self.bonds[:, 0:2]

    @atom12.setter
    def atom12(self, value):
        self.bonds[:, 0:2] = value

    @property
    def atom1(self):
        """ Return the indices of the first atoms in all bonds of **self.bonds** as 1d array"""
        return self.bonds[:, 0]

    @atom1.setter
    def atom1(self, value):
        self.bonds[:, 0] = value

    @property
    def atom2(self):
        """ Return the indices of the second atoms in all bonds of **self.bonds** as 1d array. """
        return self.bonds[:, 1]

    @atom2.setter
    def atom2(self, value):
        self.bonds[:, 1] = value

    @property
    def order(self):
        """ Return the bond orders for all bonds in **self.bonds** as 1d array. """
        return self.bonds[:, 2] / 10.0

    @order.setter
    def order(self, value):
        self.bonds[:, 2] = value * 10

    @property
    def x(self):
        """ Return the x coordinates for all atoms in **self.coords** as 2d array. """
        return self[:, :, 0]

    @property
    def y(self):
        """ Return the y coordinates for all atoms in **self.coords** as 2d array. """
        return self[:, :, 1]

    @property
    def z(self):
        """ Return the z coordinates for all atoms in **self.coords** as 2d array. """
        return self[:, :, 2]

    @property
    def symbol(self):
        """ Return the atomic symbols of all atoms in **self.atoms** as 1d array. """
        return self._get_atomic_property('symbol')

    @property
    def atnum(self):
        """ Return the atomic numbers of all atoms in **self.atoms** as 1d array. """
        return self._get_atomic_property('atnum')

    @property
    def mass(self):
        """ Return the atomic masses of all atoms in **self.atoms** as 1d array. """
        return self._get_atomic_property('mass')

    @property
    def radius(self):
        """ Return the atomic radii of all atoms in **self.atoms** as 1d array. """
        return self._get_atomic_property('radius')

    @property
    def connectors(self):
        """ Return the atomic connectors of all atoms in **self.atoms** as 1d array. """
        return self._get_atomic_property('connectors')

    def _get_atomic_property(self, prop='symbol'):
        """ Take **self.atoms** and return an (concatenated) array of a specific property associated
        with an atom type. Values are sorted by their indices.

        :parameter str prop: The to be returned property. Accepted values:
            **symbol**, **atnum**, **mass**, **radius** or **connectors**.
            See the |PeriodicTable|_ module of PLAMS for more details.
        :return: A dictionary with atomic indices as keys and atomic symbols as values.
        :rtype: |np.array|_ [|np.float64|_, |str|_ or |np.int64|_].
        """
        # Interpret the **values** argument
        prop_dict = {
            'symbol': lambda x: x,
            'radius': PeriodicTable.get_radius,
            'atnum': PeriodicTable.get_atomic_number,
            'mass': PeriodicTable.get_mass,
            'connectors': PeriodicTable.get_connectors
        }

        # Create a concatenated lists of the keys and values in **self.atoms**
        prop_list = []
        for at in self.atoms:
            at_prop = prop_dict[prop](at)
            prop_list += [at_prop for _ in self.atoms[at]]

        # Sort and return
        idx_gen = itertools.chain.from_iterable(self.atoms.values())
        return np.array([prop for _, prop in sorted(zip(idx_gen, prop_list))])

    """ ##################################  Magic methods  #################################### """

    def copy(self, order='C'):
        ret = super().copy(order)
        for key, value in vars(self).items():
            try:
                setattr(ret, key, value.copy())
            except AttributeError:
                setattr(ret, key, None)
        return ret

    def __copy__(self):
        return self.copy(order='K')

    def __str__(self):
        ret = 'Atomic coordinates:\n'
        ret += super().__str__()
        ret += '\n'

        # Convert atomic symbols
        ret += '\nAtomic symbols & indices:\n'
        if self.atoms is not None:
            for at in self.atoms:
                ret += str(at) + ':\t['
                if len(self.atoms[at]) <= 11:
                    for i in self.atoms[at][0:-2]:
                        ret += '{: <5.5}'.format(str(i))
                else:
                    for i in self.atoms[at][0:5]:
                        ret += '{: <5.5}'.format(str(i))
                    ret += '{: <5.5}'.format('...')
                    for i in self.atoms[at][-6:-2]:
                        ret += '{: <5.5}'.format(str(i))
                ret += '{: <4.4}'.format(str(self.atoms[at][-1])) + ']\n'
        else:
            ret += str(None) + '\n'

        ret += '\nBonds:\n' + str(self.bonds) + '\n'
        ret += '\nProperties:\n' + str(self.properties) + '\n'
        return ret
