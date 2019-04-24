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
        obj = np.asarray(coords).view(cls)
        _MultiMolecule._sanitize_arg(obj)
        atoms, bonds, properties = _MultiMolecule._sanitize_kwarg(atoms, bonds, properties)

        # Set attributes
        obj.atoms = _MultiMolecule._sanitize_atoms(atoms)
        obj.bonds = _MultiMolecule._sanitize_bonds(bonds)
        obj.properties = _MultiMolecule._sanitize_properties(properties)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.atoms = getattr(obj, 'atoms', None)
        self.bonds = getattr(obj, 'bonds', None)
        self.properties = getattr(obj, 'properties', None)

    @staticmethod
    def _sanitize_coords(coords):
        """ A function for sanitizing the 'coords' arguments in :meth:`_MultiMolecule.__new__`. """
        shape_error = "The 'coords' argument expects a 'm*n*3' list-like object with.\
                       The following shape was observed: '{}'"
        dtype_error = "The 'coords' argument expects a list-like object consisting exclusively of integers.\
                       The following type was observed: '{}'"

        if not coords.ndim == 3 or coords.shape[2] == 3:
            shape = ''.join('*' + str(i) for i in coords.shape)[1:]
            raise ValueError(shape_error.format(shape))
        if not isinstance(coords[0, 0, 0], np.float):
            ar_type = coords[0, 0, 0].__class__.__name__
            raise ValueError(dtype_error.format(ar_type))

    @staticmethod
    def _sanitize_bonds(bonds):
        """ A function for sanitizing the 'bonds' arguments in :meth:`_MultiMolecule.__new__`. """
        shape_error = "The 'bonds' argument expects a 2-dimensional list-like object, \
                       a {:d}-dimensional object was supplied"

        if bonds is not None:
            bonds = np.asarray(bonds, dtype=int)
            if not bonds.ndim == 2:
                raise ValueError(shape_error.format(bonds.ndim))
        return bonds

    @staticmethod
    def _sanitize_atoms(atoms):
        """ A function for sanitizing the 'atoms' arguments in :meth:`_MultiMolecule.__new__`. """
        type_error = "The 'atoms' argument expects a 'dict' object; a '{}' object was supplied"

        if atoms is None:
            return {}
        else:
            if not isinstance(atoms, dict):
                raise TypeError(type_error.format(atoms.__class__.__name__))
            return atoms

    @staticmethod
    def _sanitize_properties(properties):
        """ A function for sanitizing the 'properties' arguments in :meth:`_MultiMolecule.__new__`. """
        type_error = "The 'properties' argument expects a 'dict' object; a '{}' object was supplied"

        if properties is None:
            return Settings({'atom': None, 'bond': None, 'Molecule': None})
        else:
            if not isinstance(properties, dict):
                raise TypeError(type_error.format(properties.__class__.__name__))
            return Settings(properties)

    """ ##############################  plams-based properties  ############################### """

    @property
    def atom12(self):
        """ Return the indices of the atoms for all bonds in **self.bonds** as 2D array"""
        return self.bonds[:, 0:2]

    @atom12.setter
    def atom12(self, value):
        self.bonds[:, 0:2] = value

    @property
    def atom1(self):
        """ Return the indices of the first atoms in all bonds of **self.bonds** as 1D array"""
        return self.bonds[:, 0]

    @atom1.setter
    def atom1(self, value):
        self.bonds[:, 0] = value

    @property
    def atom2(self):
        """ Return the indices of the second atoms in all bonds of **self.bonds** as 1D array. """
        return self.bonds[:, 1]

    @atom2.setter
    def atom2(self, value):
        self.bonds[:, 1] = value

    @property
    def order(self):
        """ Return the bond orders for all bonds in **self.bonds** as 1D array. """
        return self.bonds[:, 2] / 10.0

    @order.setter
    def order(self, value):
        self.bonds[:, 2] = value * 10

    @property
    def x(self):
        """ Return the x coordinates for all atoms in **self** as 2D array. """
        return self[:, :, 0]

    @x.setter
    def x(self, value):
        self[:, :, 0] = value

    @property
    def y(self):
        """ Return the y coordinates for all atoms in **self** as 2D array. """
        return self[:, :, 1]

    @y.setter
    def y(self, value):
        self[:, :, 1] = value

    @property
    def z(self):
        """ Return the z coordinates for all atoms in **self** as 2D array. """
        return self[:, :, 2]

    @z.setter
    def z(self, value):
        self[:, :, 2] = value

    @property
    def symbol(self):
        """ Return the atomic symbols of all atoms in **self.atoms** as 1D array. """
        return self._get_atomic_property('symbol')

    @property
    def atnum(self):
        """ Return the atomic numbers of all atoms in **self.atoms** as 1D array. """
        return self._get_atomic_property('atnum')

    @property
    def mass(self):
        """ Return the atomic masses of all atoms in **self.atoms** as 1D array. """
        return self._get_atomic_property('mass')

    @property
    def radius(self):
        """ Return the atomic radii of all atoms in **self.atoms** as 1d array. """
        return self._get_atomic_property('radius')

    @property
    def connectors(self):
        """ Return the atomic connectors of all atoms in **self.atoms** as 1D array. """
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

    def copy(self, order='C', copy_attr=True):
        """ Return a copy of the MultiMolecule object.

        :parameter str order: Controls the memory layout of the copy.
            see np.ndarray.copy()_ for more details.
        :parameter bool copy_attr: Whether or not the attributes of **self** should returned as
            copies or views.
        :return: A copy of **self**.
        :rtype: |FOX.MultiMolecule|_
        .. _np.ndarray.copy(): https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.copy.html
        """
        ret = super().copy(order)
        if not copy_attr:
            return ret

        # Copy attributes
        for key, value in vars(self).items():
            try:
                setattr(ret, key, value.copy())
            except AttributeError:
                setattr(ret, key, None)
        return ret

    def __copy__(self):
        return self.copy(order='K', copy_attr=False)

    def __deepcopy__(self, memo):
        return self.copy(order='K', copy_attr=True)

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
