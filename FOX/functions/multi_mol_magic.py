""" A Module for setting up the magic methods of the MultiMolecule class. """

__all__ = []

import numpy as np

from scm.plams.core.settings import Settings

from .read_xyz import read_multi_xyz


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

    def _sanitize_other(self, other):
        """ A function for sanitizing the argument **other**, an argument which appears in many
        magic methods. """
        # Validate object type and shape
        try:
            # Check for arrays
            assert other.shape[-1] == 3

            # Broadcast (if necessary) and return
            if other.ndim == 2:
                other = other[None, :, :]
            elif other.ndim == 1:
                other = other[None, None, :]
            return other
        except AttributeError:
            # Assume other is a float or int
            return other

    def _set_shape(self, value): self.coords.shape = value
    def _get_shape(self): return self.coords.shape
    shape = property(_get_shape, _set_shape)

    def _set_dtype(self, value): self.coords.dtype = value
    def _get_dtype(self): return self.coords.dtype
    dtype = property(_get_dtype, _set_dtype)

    def _get_dtype(self): return self.coords.flags
    flags = property(_get_dtype, _set_dtype)

    def _get_dtype(self): return self.coords.ndim
    ndim = property(_get_dtype)

    def _get_dtype(self): return self.coords.nbytes
    nbytes = property(_get_dtype)

    """ ############################  Comparison magic methods  ############################### """

    def __eq__(self, other):
        other = self._sanitize_other(other)
        return self.coords == other

    def __ne__(self, other):
        other = self._sanitize_other(other)
        return self.coords != other

    def __lt__(self, other):
        other = self._sanitize_other(other)
        return self.coords < other

    def __gt__(self, other):
        other = self._sanitize_other(other)
        return self.coords > other

    def __le__(self, other):
        other = self._sanitize_other(other)
        return self.coords <= other

    def __ge__(self, other):
        other = self._sanitize_other(other)
        return self.coords >= other

    """ ########################### Unary operators and functions  ############################ """

    def __min__(self, axis=None):
        return self.coords.nanmin(axis)

    def __max__(self, axis=None):
        return self.coords.nanmax(axis)

    def __pos__(self):
        return self.coords

    def __neg__(self):
        return -1 * self.coords

    def __abs__(self):
        return np.abs(self.coords)

    def __round__(self, decimals=0):
        return np.round(self.coords, decimals)

    def __floor__(self):
        return np.floor(self.coords)

    def __ceil__(self):
        return np.ceil(self.coords)

    def __trunc__(self):
        return np.trunc(self.coords)

    """ ##########################  Normal arithmetic operators  ############################## """

    def __add__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = self.coords + other
        return ret

    def __sub__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = self.coords - other
        return ret

    def __mul__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = self.coords * other
        return ret

    def __matmul__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = self.coords @ other
        return ret

    def __floordiv__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = self.coords // other
        return ret

    def __div__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = self.coords / other
        return ret

    def __mod__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = self.coords % other
        return ret

    def __divmod__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = np.divmod(self.coords, other)
        return ret

    def __pow__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = self.coords**other
        return ret

    """ ##########################  Reflected arithmetic operators  ########################### """

    def __rsub__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = other - self.coords
        return ret

    def __rfloordiv__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = other // self.coords
        return ret

    def __rdiv__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = other / self.coords
        return ret

    def __rmod__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = other % self.coords
        return ret

    def __rdivmod__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = np.divmod(other, self.coords)
        return ret

    def __rpow__(self, other):
        other = self._sanitize_other(other)
        ret = self.copy()
        ret.coords = other**self.coords
        return ret

    """ ##############################  Augmented assignment  ################################# """

    def __iadd__(self, other):
        other = self._sanitize_other(other)
        self.coords += other
        return self

    def __isub__(self, other):
        other = self._sanitize_other(other)
        self.coords -= other
        return self

    def __imul__(self, other):
        other = self._sanitize_other(other)
        self.coords *= other
        return self

    def __imatmul__(self, other):
        other = self._sanitize_other(other)
        self.coords = self.coords @ np.swapaxes(other, -2, -1)
        return self

    def __ifloordiv__(self, other):
        other = self._sanitize_other(other)
        self.coords //= other
        return self

    def __idiv__(self, other):
        other = self._sanitize_other(other)
        self.coords /= other
        return self

    def __imod__(self, other):
        other = self._sanitize_other(other)
        self.coords %= other
        return self

    def __ipow__(self, other):
        other = self._sanitize_other(other)
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
            ret += 'Atom1 Atom2 Bond order'
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
        self.coords[key] = value

    def __delitem__(self, key):
        raise ValueError('cannot delete array elements')

    def __iter__(self):
        return iter(self.coords)

    def __reversed__(self, axis=0):
        ret = self.copy()
        ret.coords = np.flip(self.coords, axis=axis)
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
