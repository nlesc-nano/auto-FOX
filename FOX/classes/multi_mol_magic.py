""" A Module for setting up the magic methods of the MultiMolecule class. """

__all__ = []

import itertools

import numpy as np

from scm.plams import PeriodicTable
from scm.plams.core.settings import Settings

from ..functions.read_xyz import read_multi_xyz


class _MultiMolecule:
    """ A class for handling the magic methods of *MultiMolecule*. """
    def __init__(self, coords=None, atoms=None, bonds=None, properties=None, filename=None,
                 inputformat=None):
        # Sanitize arguments
        coords, atoms, bonds, properties = self._sanitize_init(coords, atoms, bonds, properties)

        # Set attributes
        self.coords = coords
        self.atoms = atoms
        self.bonds = bonds
        self.properties = properties

        # If **filename** is not *None*, grab coordinates from **filename**
        if filename is not None:
            format_dict = {'xyz': read_multi_xyz}

            # Try to infere the inputformat from **filename** if **inputformat** is *None*
            if inputformat is None:
                try:
                    inputformat = filename.rsplit('.', 1)[-1]
                except AttributeError:
                    error = 'MultiMolecule: Inputformat is None and no extension was found in '
                    error += '**filename**'
                    raise TypeError(error)

            # Read **filename** if its inputformat is supported
            if inputformat.lower().rsplit('.', 1)[-1] in format_dict:
                self._read(filename, inputformat, format_dict)
            else:
                error = 'MultiMolecule: No functions are available for handling ' + str(inputformat)
                error += ' files, currently supported file types are: '
                error += str(list(format_dict.keys()))
                raise KeyError(error)

    def _read(self, filename, inputformat, format_dict):
        """ A function for reading coordinates from external files. """
        coords, atoms = format_dict[inputformat](filename)
        self.coords = coords
        self.atoms = atoms

    def _sanitize_init(self, coords=None, atoms=None, bonds=None, properties=None):
        """ A function for sanitizing the arguments of __init__(). """
        # If coords is another MultiMolecule object, create a shallow copy and return
        if isinstance(coords, _MultiMolecule):
            properties = coords.properties
            bonds = coords.bonds
            atoms = coords.atoms
            coords = coords.coords
            return coords, atoms, bonds, properties

        # Sanitize coords
        assert coords is None or isinstance(coords, np.ndarray)
        if coords is not None:
            if coords.ndim == 2:
                coords = coords[None, :, :]
            elif not coords.ndim == 3:
                raise TypeError('MultiMolecule: **coords** should be None or a 2D/3D array.')

        # Sanitize atoms & properties
        assert atoms is None or isinstance(atoms, dict)
        assert properties is None or isinstance(properties, dict)
        if isinstance(atoms, dict):
            atoms = Settings(atoms)
        properties = properties or Settings()

        # Sanitize bonds
        assert bonds is None or isinstance(bonds, np.ndarray)
        if bonds is not None:
            assert bonds.ndim == 2 and bonds.shape[1] in (2, 3)
            assert bonds.dtype.type in (np.int8, np.int16, np.int32, np.int64)
            if bonds.shape[1] == 2:
                bonds = np.hstack((bonds, np.zeros(len(bonds), dtype=int)[:, None]))

        return coords, atoms, bonds, properties

    """ ##############################  plams-based properties  ############################### """

    def _set_atom1(self, value): self.bonds[:, 0] = value
    def _get_atom1(self): return self.bonds[:, 0]
    atom1 = property(_get_atom1, _set_atom1)

    def _set_atom2(self, value): self.bonds[:, 1] = value
    def _get_atom2(self): return self.bonds[:, 1]
    atom2 = property(_get_atom2, _set_atom2)

    def _set_order(self, value): self.bonds[:, 2] = value
    def _get_order(self): return self.bonds[:, 2]
    order = property(_get_order, _set_order)

    def _set_x(self, value): self.coords[:, :, 0] = value
    def _get_x(self): return self.coords[:, :, 0]
    x = property(_get_x, _set_x)

    def _set_y(self, value): self.coords[:, :, 1] = value
    def _get_y(self): return self.coords[:, :, 1]
    y = property(_get_y, _set_y)

    def _set_z(self, value): self.coords[:, :, 2] = value
    def _get_z(self): return self.coords[:, :, 2]
    z = property(_get_z, _set_z)

    def _get_symbol(self): return self._get_atomic_property('symbol')
    symbol = property(_get_symbol)

    def _get_atnum(self): return self._get_atomic_property('atnum')
    atnum = property(_get_atnum)

    def _get_mass(self): return self._get_atomic_property('mass')
    mass = property(_get_mass)

    def _get_radius(self): return self._get_atomic_property('radius')
    radius = property(_get_radius)

    def _get_connectors(self): return self._get_atomic_property('connectors')
    connectors = property(_get_connectors)

    def _get_atomic_property(self, prop='symbol'):
        """ Take **self.atoms** and return an (concatenated) array of a specific property associated
        with an atom type. Values are sorted by their indices.

        :parameter str prop: The to be returned property. Accepted values:
            **symbol**, **atnum**, **mass**, **radius** or **connectors**.
            See the |PeriodicTable|_ module of PLAMS for more details.
        :return: A dictionary with atomic indices as keys and atomic symbols as values.
        :rtype: |np.array|_ [|np.float64|_, |str|_ or |np.int64|_].
        """
        def get_symbol(symbol): return symbol

        # Interpret the **values** argument
        prop_dict = {
                'symbol': get_symbol,
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
        idx_list = itertools.chain.from_iterable(self.atoms.values())
        return np.array([prop for _, prop in sorted(zip(idx_list, prop_list))])

    """ ############################  np.ndarray-based properties  ############################ """

    def _set_shape(self, value): self.coords.shape = value
    def _get_shape(self): return self.coords.shape
    shape = property(_get_shape, _set_shape)

    def _set_dtype(self, value): self.coords.dtype = value
    def _get_dtype(self): return self.coords.dtype
    dtype = property(_get_dtype, _set_dtype)

    def _get_flags(self): return self.coords.flags
    flags = property(_get_flags)

    def _get_ndim(self): return self.coords.ndim
    ndim = property(_get_ndim)

    def _get_nbytes(self): return self.coords.nbytes
    nbytes = property(_get_nbytes)

    def _transpose(self): return np.swapaxes(self.coords, 1, 2)
    T = property(_transpose)

    """ ############################  Comparison magic methods  ############################### """

    def __eq__(self, other):
        return self.coords == other

    def __ne__(self, other):
        return self.coords != other

    def __lt__(self, other):
        return self.coords < other

    def __gt__(self, other):
        return self.coords > other

    def __le__(self, other):
        return self.coords <= other

    def __ge__(self, other):
        return self.coords >= other

    """ ########################### Unary operators and functions  ############################ """

    def __pos__(self):
        return self.coords

    def __neg__(self):
        return -1 * self.coords

    def __abs__(self):
        return np.abs(self.coords)

    def __round__(self, ndigits=0):
        return np.round(self.coords, ndigits)

    def __floor__(self):
        return np.floor(self.coords)

    def __ceil__(self):
        return np.ceil(self.coords)

    def __trunc__(self):
        return np.trunc(self.coords)

    """ ##########################  Normal arithmetic operators  ############################## """

    def __add__(self, other):
        ret = self.__copy__()
        ret.coords = self.coords + other
        return ret

    def __sub__(self, other):
        ret = self.__copy__()
        ret.coords = self.coords - other
        return ret

    def __mul__(self, other):
        ret = self.__copy__()
        ret.coords = self.coords * other
        return ret

    def __matmul__(self, other):
        ret = self.__copy__()
        ret.coords = self.coords @ other
        return ret

    def __floordiv__(self, other):
        ret = self.__copy__()
        ret.coords = self.coords // other
        return ret

    def __truediv__(self, other):
        ret = self.__copy__()
        ret.coords = self.coords / other
        return ret

    def __mod__(self, other):
        ret = self.__copy__()
        ret.coords = self.coords % other
        return ret

    def __divmod__(self, other):
        ret = self.__copy__()
        ret.coords = np.divmod(self.coords, other)
        return ret

    def __pow__(self, other):
        ret = self.__copy__()
        ret.coords = self.coords**other
        return ret

    """ ##########################  Reflected arithmetic operators  ########################### """

    def __rsub__(self, other):
        ret = self.__copy__()
        ret.coords = other - self.coords
        return ret

    def __rfloordiv__(self, other):
        ret = self.__copy__()
        ret.coords = other // self.coords
        return ret

    def __rdiv__(self, other):
        ret = self.__copy__()
        ret.coords = other / self.coords
        return ret

    def __rmod__(self, other):
        ret = self.__copy__()
        ret.coords = other % self.coords
        return ret

    def __rdivmod__(self, other):
        ret = self.__copy__()
        ret.coords = np.divmod(other, self.coords)
        return ret

    def __rpow__(self, other):
        ret = self.__copy__()
        ret.coords = other**self.coords
        return ret

    """ ##############################  Augmented assignment  ################################# """

    def __iadd__(self, other):
        self.coords += other
        return self

    def __isub__(self, other):
        self.coords -= other
        return self

    def __imul__(self, other):
        self.coords *= other
        return self

    def __imatmul__(self, other):
        self.coords = self.coords @ other
        return self

    def __ifloordiv__(self, other):
        self.coords //= other
        return self

    def __itruediv__(self, other):
        self.coords /= other
        return self

    def __imod__(self, other):
        self.coords %= other
        return self

    def __ipow__(self, other):
        self.coords **= other
        return self

    """ ##########################  Type conversion magic methods  ############################ """

    def __str__(self):
        # Convert atomic coordinates
        ret = 'Atomic coordinates:\n'
        if self.coords is not None:
            ret += str(self.coords) + '\n'
        else:
            ret += str(None) + '\n'
        ret += '\n'

        # Convert atomic symbols
        ret += 'Atomic symbols & indices:\n'
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
        ret += '\n'

        # Convert bonds
        ret += 'Bond indices and orders:\n'
        if self.bonds is not None:
            ret += 'Atom1 Atom2 Order\n'
            for at1, at2, order in self.bonds:
                ret += '[' + '{: <5.5}'.format(str(at1)) + '{: <5.5}'.format(str(at2)) + '] '
                ret += str(order) + '\n'
        else:
            ret += str(None) + '\n'
        ret += '\n'

        # Convert properties
        ret += 'Properties:\n'
        if self.properties:
            ret += str(self.properties) + '\n'
        else:
            ret += str(None) + '\n'

        return ret

    """ ################################  Custom Sequences  ################################### """

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, key):
        if not isinstance(key, str):
            return self.coords[key]
        return self.atoms[key]

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            self.coords[key] = value
        else:
            self.atoms[key] = value

    def __delitem__(self, key):
        if not isinstance(key, str):
            del self.atoms[key]
        else:
            raise ValueError('cannot delete array elements')

    def __iter__(self):
        return iter(self.coords)

    def __reversed__(self):
        ret = self.__copy__()
        ret.coords = np.flip(self.coords, axis=0)
        return ret

    def __contains__(self, item):
        return item in self.coords

    """ ###################################  Copy  ############################################ """

    def __copy__(self, subset=None):
        """ Magic method, create and return a new *MultiMolecule* and fill its attributes with
        views of their respective counterparts in **self**.

        :parameter subset: Copy a subset of attributes from **self**; if *None*, copy all
            attributes. Accepts one or more of the following values as strings: *properties*,
            *atoms* and/or *bonds*.
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        """
        subset = subset or ('atoms', 'bonds', 'properties')
        ret = _MultiMolecule()

        # Copy atoms
        if 'atoms' in subset:
            ret.coords = self.coords
            ret.atoms = self.atoms

        # Copy bonds
        if 'bonds' in subset:
            ret.bonds = self.bonds

        # Copy properties
        if 'properties' in subset:
            ret.properties = self.properties

        return ret

    def __deepcopy__(self, subset=None):
        """ Magic method, create and return a deep copy of **self**.

        :parameter subset: Deepcopy a subset of attributes from **self**; if *None*, copy all
            attributes. Accepts one or more of the following values as string:
            *properties*, *atoms* and/or *bonds*.
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        """
        subset = subset or ('atoms', 'bonds', 'properties')
        ret = self.__copy__(subset=subset)

        # Deep copy atoms
        if 'atoms' in subset:
            try:
                ret.coords = self.coords.copy()
                ret.atoms = self.atoms.copy()
            except AttributeError:
                pass

        # Deep copy bonds
        if 'bonds' in subset and self.bonds is not None:
            try:
                ret.bonds = self.bonds.copy()
            except AttributeError:
                pass

        # Deep copy properties
        if 'properties' in subset:
            try:
                ret.properties = self.properties.copy()
            except AttributeError:
                pass

        return ret
