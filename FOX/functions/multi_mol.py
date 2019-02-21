""" A Module for the MultiMolecule class. """

__all__ = ['MultiMolecule']

from itertools import chain

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scm.plams.mol.atom import Atom
from scm.plams.mol.bond import Bond
from scm.plams.mol.molecule import Molecule
from scm.plams.tools.periodic_table import PeriodicTable

from .rdf import (get_rdf, get_rdf_lowmem, get_rdf_df)
from .multi_mol_magic import _MultiMolecule


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
        atom_subset = self.subset_to_idx(atom_subset)
        mol = self.as_Molecule(mol_subset=0, atom_subset=atom_subset)[0]
        mol.guess_bonds()
        self.from_Molecule(mol, subset='bonds')

    def remove_random_coords(self, p=0.5, inplace=False):
        """ Remove random molecules from **self.coords**.
        For each molecule, the probability of removal is equal to **p**.
        Performs an inplace modification of **self.coords** if **return_coords** is *False*.

        :parameter float p: The probability of remove a 2D slice from **self.coords**.
            Accepts values between 0.0 (0%) and 1.0 (100%).
        :parameter bool inplace: If *False*, return a view of the gathered 2D frames.
            If *True*, perform an inplace modification **self.coords**, replacing it with a
            view of the randomly gathered molecules.
        """
        if p <= 0.0 or p >= 1.0:
            raise IndexError('The probability, p, must be larger than 0.0 and smaller than 1.0')
        elif self.shape[0] == 1:
            raise IndexError('Grabbing random 2D slices from a 2D array makes no sense')

        size = 1 or int(self.shape[0] / p)
        idx_range = np.arange(self.shape[0])
        idx = np.random.choice(idx_range, size)
        if not inplace:
            return self[idx]
        self.coords = self[idx]

    """ ################################## Root Mean Squared ################################## """

    def init_rmsd(self, other, mol_subset=None, atom_subset=None):
        """ """
        # Figure out if the RMSD should be calculated as a single or multiple series
        try:
            if isinstance(atom_subset[0], str) and not isinstance(atom_subset, str):
                loop = True
            else:
                loop = False
        except TypeError:
            loop = False

        # Get the RMSD
        if loop:
            rmsd = np.array([self.get_rmsd(other, mol_subset, at) for at in atom_subset]).T
        else:
            rmsd = self.get_rmsd(other, mol_subset, atom_subset)

        # Construct arguments for the dataframe
        if loop:
            columns = atom_subset
        elif isinstance(atom_subset, str):
            columns = [atom_subset]
        else:
            columns = ['RMSD']
        index = mol_subset or np.arange(0, self.shape[0])

        # Create, fill and return a dataframe with the RMSD
        df = pd.DataFrame(data=rmsd, index=index, columns=columns)
        df.columns.name = 'RMSD  /  Ångström'
        df.index.name = 'XYZ frame'
        return df

    def init_rmsf(self, mol_subset=None, atom_subset=None):
        """ """
        # Figure out if the RMSD should be calculated as a single or multiple series
        try:
            if isinstance(atom_subset[0], str) and not isinstance(atom_subset, str):
                loop = True
            else:
                loop = False
        except TypeError:
            loop = False

        # Get the RMSF
        if loop:
            rmsf = np.array([self.get_rmsf(mol_subset, at) for at in atom_subset]).T
        else:
            rmsf = self.get_rmsf(mol_subset, atom_subset)

        # Construct arguments for the dataframe
        index = np.arange(0, self.shape[1])
        if loop:
            columns = atom_subset
            data = np.empty((rmsf.shape[0], index.shape[0]))
            data[:, :] = np.nan
            for i, (at, j) in enumerate(zip(list(atom_subset), rmsf)):
                idx = self.subset_to_idx(at)
                data[i, idx] = j
            data = data.T
        elif isinstance(atom_subset, str):
            columns = [atom_subset]
            data = np.empty((index.shape[0]))
            data[:] = np.nan
            idx = self.subset_to_idx(atom_subset)
            data[idx] = rmsf
        else:
            columns = ['RMSF']
            data = rmsf

        # Create, fill and return a dataframe with the RMSF
        df = pd.DataFrame(data=data, index=index, columns=columns)
        df.columns.name = 'RMSF  /  Ångström'
        df.index.name = 'Atomic index'
        return df

    def get_rmsd(self, other, mol_subset=None, atom_subset=None):
        """ Calculate the root mean square fluctuation (RMSF) between **self.coords** and
        **other**. Returns a dataframe with the RMSD as a function of the XYZ frame numbers. """
        # Prepare slices
        i = mol_subset or slice(0, self.shape[0])
        j = self.subset_to_idx(atom_subset) or slice(0, self.shape[1])

        # Check if the shapes of self and other match
        if self[i, j, :].shape == other[i, j, :].shape:
            other = other[i, j, :]

        # Calculate and return the RMSD per molecule in **self.coords**
        dist = np.linalg.norm(self[i, j, :] - other, axis=2)
        return np.sqrt(np.einsum('ij,ij->i', dist, dist) / dist.shape[1])

    def get_rmsf(self, mol_subset=None, atom_subset=None):
        """ Calculate the root mean square fluctuation (RMSF) of **self.coords**.
        Returns a dataframe as a function of atomic indices. """
        # Prepare slices
        i = mol_subset or slice(0, self.shape[0])
        j = self.subset_to_idx(atom_subset) or slice(0, self.shape[1])

        # Calculate the RMSF per molecule in **self.coords**
        mean_coords = np.average(self[i, j, :], axis=0)[None, :, :]
        displacement = np.linalg.norm(self[i, j, :] - mean_coords, axis=2)
        return np.linalg.norm(displacement, axis=0)

    def reset_origin(self, mol_subset=None, atom_subset=None, inplace=True):
        """ Set the origin to the center of mass of **self.coords**. Performs in inplace update
        of **self.coords** if **inplace** is *True*. """
        center_of_mass = self.get_center_of_mass(mol_subset, atom_subset)
        if not inplace:
            return self - center_of_mass[:, None, :]
        self -= center_of_mass[:, None, :]

    def get_center_of_mass(self, mol_subset=None, atom_subset=None):
        """ Return the center of mass of *m* molecules as an *m*3* array. """
        # Prepare a dictionary with atomic masses
        mass_dict = {}
        for at in self.atoms:
            mass = PeriodicTable.get_mass(at)
            for i in self.atoms[at]:
                try:
                    mass_dict[i].append(mass)
                except KeyError:
                    mass_dict[i] = [mass]

        # Prepare slices
        i = mol_subset or slice(0, self.shape[0])
        j = self.subset_to_idx(atom_subset) or slice(0, self.shape[1])

        # Get the center of mass
        mass = np.array([mass_dict[i] for i in atom_subset])[None, :]
        weighted_coords = self[i, j, :] * mass[:, None]
        center = weighted_coords.sum(axis=2)
        return center / mass.sum(axis=1)

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
        i = mol_subset, self.subset_to_idx(atom_subset[0]) or slice(0, self.shape[1])
        j = mol_subset, self.subset_to_idx(atom_subset[1]) or slice(0, self.shape[1])

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
        i = mol_subset, self.subset_to_idx(atom_subset[0]) or slice(0, self.shape[1])
        j = mol_subset, self.subset_to_idx(atom_subset[1]) or slice(0, self.shape[1])
        k = mol_subset, self.subset_to_idx(atom_subset[2]) or slice(0, self.shape[1])

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

    def subset_to_idx(self, arg):
        """ Grab and return a list of indices from **self.atoms**.
        Return *at* if it is *None*, an *int* or iterable container consisting of *int*. """
        if arg is None:
            return None
        elif isinstance(arg, int) or isinstance(arg[0], int):
            return arg
        elif isinstance(arg, str):
            return self.atoms[arg]
        elif isinstance(arg[0], str):
            return list(chain.from_iterable(self.atoms[i] for i in arg))
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
        mol_subset = mol_subset or np.arange(1, self.shape[0])
        atom_subset = self.subset_to_idx(atom_subset)
        atom_subset = atom_subset or list(chain.from_iterable(self.atoms.values()))

        # Construct a template molecule and fill it with atoms
        assert self.coords is not None
        assert self.atoms is not None
        mol_template = Molecule()
        mol_template.properties = self.properties.copy()
        for i, symbol in enumerate(atom_subset):
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
        for i, xyz in zip(self, mol_subset):
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
