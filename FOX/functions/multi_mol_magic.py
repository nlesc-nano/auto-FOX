import numpy as np

from scm.plams.core.settings import Settings


class _MultiMolecule:
    def __init__(self, coords=None, atoms=None, bonds=None, bond_orders=None, properties=None):
        # Sanitize coords
        assert coords is None or isinstance(coords, np.ndarray)
        if coords is not None:
            if len(coords.shape) == 2:
                coords = coords[None, :, :]
            elif not len(coords.shape) == 3:
                raise TypeError('MultiMolecule: **coords** should be None or a 2D/3D array.')

        # Sanitize atoms & properties
        assert atoms is None or isinstance(atoms, dict)
        assert properties is None or isinstance(properties, dict)
        atoms = atoms or {}
        properties = properties or {}

        # Sanitize bonds
        assert bonds is None or isinstance(coords, np.ndarray)
        if bonds is not None:
            assert bonds.dtype.type in (np.int8, np.int16, np.int32, np.int64)
        assert bond_orders is None or isinstance(coords, np.ndarray)
        if bond_orders is not None:
            if bond_orders.dtype.type not in (np.int8, np.int16, np.int32, np.int64):
                bond_orders = np.array(bond_orders, dtype=float)

        # Set attributes
        self.coords = coords
        self.dtype = coords.dtype
        self.shape = coords.shape
        self.atoms = Settings(atoms)
        self.bonds = bonds
        self.bond_orders = bond_orders
        self.properties = Settings(properties)

    def _sanitize_other(self, other):
        # Validate object type and shape
        assert isinstance(other, np.ndarray)
        assert other.shape[-1] == 3

        # Broadcast (if necessary) and return
        if len(other.shape) == 2:
            other = other[None, :, :]
        elif len(other.shape) == 1:
            other = other[None, None, :]
        return other

    ###############################  Comparison magic methods  ####################################

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

    ############################ Unary operators and functions  ###################################

    def __min__(self, axis=None):
        return self.coords.nanmin(axis)

    def __max__(self, axis=None):
        return self.coords.nanmax(axis)

    def __pos__(self):
        return self.coords

    def __neg__(self):
        return -1 * self.coords

    def __abs__(self, kwarg):
        return np.abs(self.coords, **kwarg)

    def __invert__(self):
        return ~self.coords

    def __round__(self, decimals=0):
        return np.round(self.coords, decimals=decimals)

    def __floor__(self):
        return np.floor(self.coords)

    def __ceil__(self):
        return np.ceil(self.coords)

    def __trunc__(self):
        return np.trunc(self.coords)

    ############################  Normal arithmetic operators  ####################################

    def __add__(self, other):
        other = self._sanitize_other(other)
        return self.coords + other

    def __sub__(self, other):
        other = self._sanitize_other(other)
        return self.coords - other

    def __mul__(self, other):
        other = self._sanitize_other(other)
        return self.coords * other

    def __matmul__(self, other):
        other = self._sanitize_other(other)
        return self.coords @ other

    def __floordiv__(self, other):
        other = self._sanitize_other(other)
        return self.coords // other

    def __div__(self, other):
        other = self._sanitize_other(other)
        return self.coords / other

    def __mod__(self, other):
        other = self._sanitize_other(other)
        return self.coords % other

    def __divmod__(self, other):
        other = self._sanitize_other(other)
        return np.divmod(self.coords, other)

    def __pow__(self, other):
        other = self._sanitize_other(other)
        return self.coords**other

    ###################################  Reflected arithmetic operators  ##########################

    def __rsub__(self, other):
        other = self._sanitize_other(other)
        return other - self.coords

    def __rfloordiv__(self, other):
        other = self._sanitize_other(other)
        return other // self.coords

    def __rdiv__(self, other):
        other = self._sanitize_other(other)
        return other / self.coords

    def __rmod__(self, other):
        other = self._sanitize_other(other)
        return other % self.coords

    def __rdivmod__(self, other):
        other = self._sanitize_other(other)
        return np.divmod(other, self.coords)

    def __rpow__(self, other):
        other = self._sanitize_other(other)
        return other**self.coords

    ################################  Augmented assignment  #######################################

    def __iadd__(self, other):
        other = self._sanitize_other(other)
        self.coords += other

    def __isub__(self, other):
        other = self._sanitize_other(other)
        self.coords -= other

    def __imul__(self, other):
        other = self._sanitize_other(other)
        self.coords *= other

    def __imatmul__(self, other):
        other = self._sanitize_other(other)
        self.coords = self.coords @ other

    def __ifloordiv__(self, other):
        other = self._sanitize_other(other)
        self.coords //= other

    def __idiv__(self, other):
        other = self._sanitize_other(other)
        self.coords /= other

    def __imod__(self, other):
        other = self._sanitize_other(other)
        self.coords %= other

    def __ipow__(self, other):
        other = self._sanitize_other(other)
        self.coords **= other

    #############################  Type conversion magic methods  #################################

    def __str__(self):
        ret = 'Atomic coordinates:\n' + str(self.coords) + '\n\n'
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
        ret += '\n'
        ret += 'Bonds and bond orders:\n'
        if self.bonds is not None and self.bond_orders is not None:
            for i, j in zip(self.bonds, self.bond_orders):
                ret += str(i) + ' ' + str(j) + '\n'
        ret += '\n'
        ret += '\nProperties:\n' + str(self.properties) + '\n'
        return ret

    ######################################  Custom Sequences  #####################################

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
        return np.flip(self.coords, axis=axis)

    def __contains__(self, item):
        return item in self.coords
