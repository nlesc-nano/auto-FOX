""" A Module for the MultiMolecule class. """

__all__ = ['MultiMolecule']

from itertools import chain

import numpy as np
from scipy.spatial.distance import cdist

from scm.plams.mol.atom import Atom
from scm.plams.mol.bond import Bond
from scm.plams.mol.molecule import Molecule
from scm.plams.core.settings import Settings

from .read_xyz import read_multi_xyz
from .rdf import (get_rdf, get_rdf_lowmem, get_rdf_df)


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
        ret.coords = self.coords @ np.swapaxes(other, -2, -1)
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


class MultiMolecule(_MultiMolecule):
    """ A class designed for handling a and manipulating large numbers of molecules. More
    specifically, different conformations of a single molecule as derived from, for example,
    an intrinsic reaction coordinate calculation (IRC) or a molecular dymanics trajectory (MD).
    The class has access to four attributes (further details are provided under parameters):

    - **coords**: The central object behind the MultiMolecule class, an *m*n*3* Numpy holding
    the cartesian coordinates of *m* molecules with *n* atoms.

    - **atoms**: None

    - **bonds**: None

    - **properties**: None

    :parameter coords: A 3D array with the cartesian coordinates of *m* molecules with *n* atoms.
    :type coords: |None|_ or *m*n*3* |np.ndarray|_ [|np.float64|_]
    :parameter atoms: A dictionary-derived object with atomic symbols as keys and matching atomic
        indices as values.
    :type atoms: |None|_ or |plams.Settings|_ (keys: |str|_, values: |list|_ [|int|_],
                                               superclass: |dict|_)
    :parameter bonds: A 2D array with indices of the atoms defining all *k* bonds (columns 1 & 2)
        and their respective bond orders (column 3).
    :type bonds: |None|_ or *k*3* |np.ndarray|_ [|np.int64|_]
    :parameter properties: A dictionary-derived object intended for storing miscellaneous
        user-defined (meta-)data. Is devoid of keys by default.
    :type properties: |plams.Settings|_ (superclass: |dict|_)
    :parameter filename: Create a **MultiMolecule** object out of a preexisting file, requiring its
        path + filename.
    :type filename: |None|_ or |str|_
    :parameter inputformat: The filetype of **filename**. Currently supported filetypes are
        limited to (multi-) .xyz files. If *None*, try to infere the filetype from **filename**.
    :type inputformat: |None|_ or |str|_
    """

    def guess_bonds(self, atom_subset=None):
        """ Guess bonds within the molecules based on atom type and inter-atomic distances.
        Bonds are guessed based on the first molecule in **self.coords**
        Performs an inplace modification of **self.bonds**

        :parameter atom_subset: A tuple of atomic symbols. Bonds are guessed between all atoms
            whose atomic symbol is in **atom_subset**. If *None*, guess bonds for all atoms in
            **self.coords**.
        :type atom_subset: |None|_ or |tuple|_ [|str|_]
        """
        atom_subset = atom_subset or self.atoms.keys()
        mol = self.as_Molecule(mol_subset=0, atom_subset=atom_subset)[0]
        mol.guess_bonds()
        self.from_Molecule(mol, subset='bonds')

    def remove_random_coords(self, p=0.5, return_coords=False):
        """ Remove random molecules from **self.coords**.
        For each molecule, the probability of removal is equal to **p**.
        Performs an inplace modification of **self.coords** if **return_coords** is *False*.

        :parameter float p: The probability of remove a 2D slice from **self.coords**.
            Accepts values between 0.0 (0%) and 1.0 (100%).
        :parameter bool return_coords: If *True*, return a view of the gathered 2D frames.
            If *False*, perform an inplace modification **self.coords**, replacing it with a
            view of the randomly gathered molecules.
        """
        if p <= 0.0 or p >= 1.0:
            raise IndexError('The probability, p, must be larger than 0.0 and smaller than 1.0')
        elif self.shape[0] == 1:
            raise IndexError('Grabbing random 2D slices from a 2D array makes no sense')

        size = 1 or int(self.shape[0] / p)
        idx_range = np.arange(self.shape[0])
        idx = np.random.choice(idx_range, size)
        if return_coords:
            return self[idx]
        self.coords = self[idx]

    """ ################################## Root Mean Squared ################################## """

    def rms(self, ref, mol_subset=None, atom_subset=None):
        ret = np.linalg.norm(self - ref, axis=(1, 2)) / self.shape[1]
        return np.linalg.norm(self - ref) / self.shape[0]

    """ ###################  Radial and Angular Distribution Functions  ####################### """

    def init_rdf(self, atom_subset=None, dr=0.05, r_max=12.0, low_mem=False):
        """ Initialize the calculation of radial distribution functions (RDFs). RDFs are calculated
        for all possible atom-pairs in **atom_subset** and returned as a dataframe.

        :parameter atom_subset: A tuple of atomic symbols. RDFs will be calculated for all
            possible atom-pairs in **atoms**. If *None*, calculate RDFs for all possible atom-pairs
            in the keys of **self.atoms**.
        :type atom_subset: |None|_ or |tuple|_ [|str|_]
        :parameter float dr: The integration step-size in Ångström, *i.e.* the distance between
            concentric spheres.
        :parameter float r_max: The maximum to be evaluated interatomic distance in Ångström.
        :parameter bool low_mem: If *True*, use a slower but more memory efficient method for
            constructing the RDFs.
        :return: A dataframe of radial distribution functions, averaged over all conformations in
            **xyz_array**. Keys are of the form: at_symbol1 + ' ' + at_symbol2 (*e.g.* 'Cd Cd').
            Radii are used as indices.
        :rtype: |pd.DataFrame|_ (keys: |str|_, values: |np.float64|_, indices: |np.float64|_).
        """
        # If **atom_subset** is None: extract atomic symbols from they keys of **self.atoms**
        atom_subset = atom_subset or tuple(self.atoms.keys())

        # Construct a list of 2-tuples containing all unique atom pairs
        atom_pairs = [(at1, at2) for i, at1 in enumerate(atom_subset) for at2 in atom_subset[i:]]

        # Construct an empty dataframe with appropiate dimensions, indices and keys
        df = get_rdf_df(dr, r_max, atom_pairs)

        # Fill the dataframe with RDF's, averaged over all conformations in **self.coords**
        kwarg1 = {'dr': dr, 'r_max': r_max}
        if low_mem:
            # Slower low memory approach
            kwarg2 = {'mol_subset': None, 'atom_subset': None}
            for i, _ in enumerate(self):
                kwarg2['frame'] = i
                for at1, at2 in atom_pairs:
                    kwarg2['atom_subset'] = (at1, at2)
                    df[at1 + ' ' + at2] += get_rdf_lowmem(self.get_dist_mat(**kwarg2), **kwarg1)

            df.iloc[0] = 0.0
            df /= self.shape[0]
        else:
            # Faster high memory approach
            kwarg2 = {'mol_subset': None, 'atom_subset': None}
            for at1, at2 in atom_pairs:
                kwarg2['atom_subset'] = (at1, at2)
                df[at1 + ' ' + at2] = get_rdf(self.get_dist_mat(**kwarg2), **kwarg1)

        return df

    def get_dist_mat(self, mol_subset=None, atom_subset=(None, None)):
        """ Create and return a distance matrix for all molecules and atoms in **self.coords**.
        Returns a 3D array.

        :parameter mol_subset: Create a distance matrix from a subset of molecules in
            **self.coords**. If *None*, create a distance matrix for all molecules in
            **self.coords**.
        :type mol_subset: |None|_ or |tuple|_ [|int|_]
        :parameter atom_subset: Create a distance matrix from a subset of atoms per molecule in
            **self.coords**. Values have to be supplied for all 2 dimensions. Atomic indices
            (on or multiple), atomic symbols (one or multiple) and *None* can be freely mixed.
            If *None*, pick all atoms from **self.coords** for that partical dimension; if an
            atomic symbol, do the same for all indices associated with that particular symbol.
        :type atom_subset: |tuple|_ [|None|_], |tuple|_ [|int|_]
        :return: A 3D distance matrix of *m* molecules, created out of two sets of *n* and
            *k* atoms.
        :return type: *m*n*k* |np.ndarray|_ [|np.float64|_].
        """
        # Define array slices
        mol_subset = mol_subset or slice(0, self.shape[0])
        i = mol_subset, self._get_idx(atom_subset[0]) or slice(0, self.shape[1])
        j = mol_subset, self._get_idx(atom_subset[1]) or slice(0, self.shape[1])

        # Slice the XYZ array
        A = self[i]
        B = self[j]

        # Create, fill and return the distance matrix
        shape = self.shape[0], A.shape[1], B.shape[1]
        ret = np.empty(shape)
        for k, (a, b) in enumerate(zip(A, B)):
            ret[k] = cdist(a, b)
        return ret

    def get_angle_mat(self, mol_subset=0, atom_subset=(None, None, None)):
        """ Create and return an angle matrix for all molecules and atoms in **self.coords**.
        Returns a 4D array.

        :parameter mol_subset: Create a distance matrix from a subset of molecules in
            **self.coords**. If *None*, create a distance matrix for all molecules in
            **self.coords**.
        :type mol_subset: |None|_ or |tuple|_ [|int|_]
       :parameter atom_subset: Create a distance matrix from a subset of atoms per molecule in
            **self.coords**. Values have to be supplied for all 3 dimensions. Atomic indices
            (on or multiple), atomic symbols (one or multiple) and *None* can be freely mixed.
            If *None*, pick all atoms from **self.coords** for that partical dimension; if an
            atomic symbol, do the same for all indices associated with that particular symbol.
        :type atom_subset: |None|_ or |tuple|_ [|str|_]
        :return: A 4D angle matrix of *m* molecules, created out of three sets of *n*, *k* and
            *l* atoms.
        :return type: *m*n*k*l* |np.ndarray|_ [|np.float64|_].
        """
        # Define array slices
        mol_subset = mol_subset or slice(0, self.shape[0])
        i = mol_subset, self._get_idx(atom_subset[0]) or slice(0, self.shape[1])
        j = mol_subset, self._get_idx(atom_subset[1]) or slice(0, self.shape[1])
        k = mol_subset, self._get_idx(atom_subset[2]) or slice(0, self.shape[1])

        # Slice and broadcast the XYZ array
        A = self[i][:, None, :]
        B = self[j][:, :, None]
        C = self[k][:, :, None]

        # Prepare the unit vectors
        kwarg1 = {'atom_subset': [atom_subset[0], atom_subset[1]], 'mol_subset': mol_subset}
        kwarg2 = {'atom_subset': [atom_subset[0], atom_subset[2]], 'mol_subset': mol_subset}
        unit_vec1 = A - B / self.get_dist_mat(**kwarg1)
        unit_vec2 = A - C / self.get_dist_mat(**kwarg2)

        # Create and return the angle matrix
        ret = np.arccos(np.einsum('ijkl,ijml->ijkl', unit_vec1, unit_vec2))
        return ret

    def _get_idx(self, at):
        """ Grab and return a list of indices from **self.atoms**.
        Return *at* if it is *None*, an *int* or iterable container consisting of *int*. """
        if at is None:
            return at
        elif isinstance(at, int) or isinstance(at[0], int):
            return at
        elif isinstance(at, str):
            return self.atoms[at]
        elif isinstance(at[0], str):
            return list(chain.from_iterable(self.atoms[i] for i in at))
        raise KeyError()

    """ #################################  Type conversion  ################################### """

    def as_Molecule(self, mol_subset=None, atom_subset=None):
        """ Convert a *MultiMolecule* object into a *list* of *plams.Molecule*.

        :parameter mol_subset: Convert a subset of molecules in **self.coords** as based on their
            indices. If *None*, convert all molecules.
        :type mol_subset: |None|_, |int|_ or |tuple|_ [|int|_]
        :parameter atom_subset: Convert a only subset of atoms within each molecule in
            **self.coords**, as based on their indices. If *None*, convert all atoms per molecule.
        :type atom_subset: |None|_, |int|_ or |tuple|_ [|int|_]
        :return: A list of *m* PLAMS molecules.
        :rtype: |list|_ [|plams.Molecule|_].
        """
        # Create a dictionary with atomic indices as keys and matching atomic symbols as values
        atom_subset = atom_subset or self.atoms.values()
        at_list = list(chain.from_iterable(atom_subset))

        # Construct a template molecule and fill it with atoms
        assert self.coords is not None
        assert self.atoms is not None
        mol_template = Molecule()
        mol_template.properties = self.properties.copy()
        for i, symbol in enumerate(at_list):
            atom = Atom(symbol=symbol)
            mol_template.add_atom(atom)

        # Fill the template molecule with bonds
        if self.bonds is not None:
            for i, j, order in self.bonds:
                if i in atom_subset and j in atom_subset:
                    at1 = mol_template[i + 1]
                    at2 = mol_template[j + 1]
                    bond = Bond(atom1=at1, atom2=at2, order=order)
                    mol_template.add_bond(bond)

        # Create copies of the template molecule; update their cartesian coordinates
        ret = []
        idx_range = mol_subset or np.arange(1, self.shape[0])
        for i, xyz in zip(self, idx_range):
            mol = mol_template.copy()
            mol.from_array(xyz)
            mol.properties.frame = i
            ret.append(mol)

        return ret

    def from_Molecule(self, mol_list, subset=None, allow_different_length=False):
        """ Convert a list of PLAMS molecules into a *MultiMolecule* object.
        Performs an inplace modification of **self**.

        :parameter mol_list: A PLAMS molecule or list of PLAMS molecules.
        :type mol_list: |Molecule|_ or |list|_ [|plams.Molecule|_]
        :parameter subset: Transfer a subset of *plams.Molecule* attributes to **self**. If *None*,
            transfer all attributes. Accepts one or more of the following values as strings:
            *properties*, *atoms* and/or *bonds*.
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        :parameter bool allow_different_length: If *True*, allow **mol_list** and **self.coords**
            to have different lengths.
        """
        if isinstance(mol_list, Molecule):
            mol_list = [mol_list]
        mol = mol_list[0]
        subset = subset or ('atoms', 'bonds', 'properties')

        if not allow_different_length:
            # Raise an error if mol_list and self.coords are of different lengths
            if len(mol_list) != self.shape[0]:
                error = 'from_Molecule: Shape mismatch, the argument mol_list is of length '
                error += str(len(mol_list)) + ' while self.coords is of length: '
                error += str(self.shape[0])
                raise IndexError(error)

        # Convert properties
        if 'properties' in subset:
            self.properties = mol.properties

        # Convert atoms
        if 'atoms' in subset:
            dummy = Molecule()
            idx = slice(0, self.shape[0])
            self.coords = np.array([dummy.as_array(atom_subset=mol.atoms[idx]) for mol in mol_list])
            for i, at in enumerate(mol.atoms[idx]):
                try:
                    self.atoms[at].append(i)
                except KeyError:
                    self.atoms[at] = [i]

        # Convert bonds
        if 'bonds' in subset:
            mol.set_atoms_id()
            self.bonds = np.zeros((len(mol.bonds), 3), dtype=int)
            for i, bond in enumerate(mol.bonds):
                self.bonds[i][2] = int(bond.order)
                self.bonds[i][0:1] = bond.at1.id, bond.at2.id
            mol.unset_atoms_id()
            if len(mol) > self.shape[0]:
                idx = np.arange(0 - len(mol))
                self.bonds = self.bonds[idx]

    def copy(self, subset=None, deepcopy=False):
        """ Create and return a new *MultiMolecule* object and fill its attributes with
        views of their respective counterparts in **self**, creating a shallow copy.

        :parameter subset: Copy a subset of attributes from **self**; if *None*, copy all
            attributes. Accepts one or more of the following values as strings: *properties*,
            *atoms* and/or *bonds*.
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        :parameter bool deepcopy: If *True*, perform a deep copy instead of a shallow copy.
        """
        if deepcopy:
            return MultiMolecule(self.__deepcopy__(subset))
        return MultiMolecule(self.__copy__(subset))

    def deepcopy(self, subset=None):
        """ Create and return a new *MultiMolecule* object and fill its attributes with
        copies of their respective counterparts in **self**, creating a deep copy.

        :parameter subset: Copy a subset of attributes from **self**; if *None*, copy all
            attributes. Accepts one or more of the following values as strings: *properties*,
            *atoms* and/or *bonds*.
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        """
        return MultiMolecule(self.__deepcopy__(subset))
