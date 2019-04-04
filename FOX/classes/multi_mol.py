""" A Module for the MultiMolecule class. """

__all__ = ['MultiMolecule']

from itertools import chain

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scm.plams import (Atom, Bond)

from .molecule_utils import Molecule
from .multi_mol_magic import _MultiMolecule
from ..functions.rdf import (get_rdf, get_rdf_lowmem, get_rdf_df)
from ..functions.utils import (serialize_array, read_param)


class MultiMolecule(_MultiMolecule):
    """ A class designed for handling a and manipulating large numbers of molecules. More
    specifically, different conformations of a single molecule as derived from, for example,
    an intrinsic reaction coordinate calculation (IRC) or a molecular dymanics trajectory (MD).
    The class has access to four attributes (further details are provided under parameters):

    :parameter coords: A 3D array with the cartesian coordinates of *m* molecules with *n* atoms.
        Stored in the **coords** attribute.
    :type coords: |None|_ or *m*n*3* |np.ndarray|_ [|np.float64|_]
    :parameter atoms: A dictionary-derived object with atomic symbols as keys and matching atomic
        indices as values. Stored in the **atoms** attribute.
    :type atoms: |None|_ or |dict|_ (keys: |str|_, values: |list|_ [|int|_])
    :parameter bonds: A 2D array with indices of the atoms defining all *k* bonds (columns 1 & 2)
        and their respective bond orders (column 3). Stored in the **bonds** attribute.
    :type bonds: |None|_ or *k*3* |np.ndarray|_ [|np.int64|_]
    :parameter properties: A Settings object (subclass of dictionary) intended for storing
        miscellaneous user-defined (meta-)data. Is devoid of keys by default. Stored in the
        **properties** attribute.
    :type properties: |plams.Settings|_ (superclass: |dict|_)
    :parameter filename: Create a **MultiMolecule** object out of a preexisting file, requiring its
        path + filename.
    :type filename: |None|_ or |str|_
    :parameter inputformat: The filetype of **filename**. Currently supported filetypes are
        limited to (multi-) .xyz files. If *None*, try to infere the filetype from **filename**.
    :type inputformat: |None|_ or |str|_
    """

    def guess_bonds(self, charge=0, atom_subset=None):
        """ Guess bonds within the molecules based on atom type and inter-atomic distances.
        Bonds are guessed based on the first molecule in **self.coords**
        Performs an inplace modification of **self.bonds**

        :parameter atom_subset: A tuple of atomic symbols. Bonds are guessed between all atoms
            whose atomic symbol is in **atom_subset**. If *None*, guess bonds for all atoms in
            **self.coords**.
        :type atom_subset: |None|_ or |tuple|_ [|str|_]
        :parameter int charge: The total charge of all atoms in **atom_subset**.
        """
        if atom_subset is None:
            atom_subset = np.arange(0, self.shape[1])
        else:
            atom_subset = np.array(sorted(self._get_atom_subset(atom_subset)))

        # Guess bonds
        mol = self.as_Molecule(mol_subset=0, atom_subset=atom_subset)[0]
        mol.guess_bonds()
        mol.fix_bond_orders()
        self.from_Molecule(mol, subset='bonds', allow_different_length=True)

        # Update indices in **self.bonds** to account for **atom_subset**
        self.atom1 = atom_subset[self.atom1]
        self.atom2 = atom_subset[self.atom2]
        self.bonds[:, 0:2].sort(axis=1)
        idx = self.bonds[:, 0:2].argsort(axis=0)[:, 0]
        self.bonds = self.bonds[idx]

    def slice(self, start=0, stop=None, step=1, inplace=False):
        """ Construct a new *MultiMolecule* by iterating through **self.coords**
        along a set interval.

        :parameter int start: Start of the interval.
        :parameter int stop: End of the interval.
        :parameter int step: Spacing between values.
        :parameter bool inplace: If *True*, perform an inplace update of **self** instead of
            returning a new *MultiMolecule* object.
        """
        if inplace:
            self.coords = self[start:stop:step]
        else:
            ret = self.deepcopy(subset=('atoms', 'bonds', 'properties'))
            ret.coords = self[start:stop:step].copy()
            return ret

    def random_slice(self, start=0, stop=None, p=0.5, inplace=False):
        """ Construct a new *MultiMolecule* by iterating through **self.coords** at random
        intervals. The probability of including a particular element is equivalent to **p**.

        :parameter int start: Start of the interval.
        :parameter int stop: End of the interval.
        :parameter float p: The probability of including each particular molecule in
            **self.coords**. Values must be between 0.0 (0%) and 1.0 (100%).
        :parameter bool inplace: If *True*, perform an inplace update of **self** instead of
            returning a new *MultiMoleule* object.
        """
        if p <= 0.0 or p >= 1.0:
            raise IndexError('The probability, p, must be larger than 0.0 and smaller than 1.0')

        stop = stop or self.shape[0]
        idx_range = np.arange(start, stop)
        size = p * len(idx_range)
        idx = np.random.choice(idx_range, size=size, replace=False)
        if inplace:
            self.coords = self[idx]
        else:
            ret = self.deepcopy(subset=('atoms', 'bonds', 'properties'))
            ret.coords = self[idx]
            return ret

    def reset_origin(self, mol_subset=None, atom_subset=None, inplace=True):
        """ Reallign all molecules in **self**, rotating and translating them, by performing a
        partial partial Procrustes superimposition. The superimposition is carried out with respect
        to the first molecule in **self**.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all *m* molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all *n* atoms per molecule
            in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :parameter bool inplace: Instead of returning the new coordinates, perform an inplace
            update of **self.coords**.
        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Remove translations
        coords = self[i, j, :] - np.mean(self[i, j, :], axis=1)[:, None, :]

        # Peform a singular value decomposition on the covariance matrix
        H = np.swapaxes(coords[0:], 1, 2) @ coords[0]
        U, S, Vt = np.linalg.svd(H)
        V, Ut = np.swapaxes(Vt, 1, 2), np.swapaxes(U, 1, 2)

        # Construct the rotation matrix
        rotmat = np.ones_like(U)
        rotmat[:, 2, 2] = np.linalg.det(V @ Ut)
        rotmat *= V@Ut

        # Return or perform an inplace update of **self.coords**
        if inplace:
            self[i, j, :] = coords @ np.swapaxes(rotmat, 1, 2)
        else:
            ret = self.deepcopy()
            ret[i, j, :] = coords @ rotmat
            return ret

    def sort(self, sort_by='symbol', reverse=False):
        """ Sort the atoms in **self.coords** and **self.atoms**, performing in inplace update.

        :parameter sort_by: The property which is to be used for sorting. Accepted values:
            **symbol** (*i.e.* alphabetical), **atnum**, **mass**, **radius** or
            **connectors**. See the PeriodicTable_ module of PLAMS for more details.
            Alternatively, a user-specified array of indices can be provided for sorting.
        :type sort_by: |str|_ or |np.ndarray|_ [|np.int64|_]
        :parameter bool reverse: Sort in reversed order.
        """
        # Create and, potentially, sort a list of indices
        if isinstance(sort_by, str):
            sort_by_array = self._get_atomic_property(prop=sort_by)
            idx_range = range(self.shape[0])
            idx_range = np.array([i for _, i in sorted(zip(sort_by_array, idx_range))])
        else:
            assert sort_by.shape[0] == self.shape[1]
            idx_range = sort_by

        # Reverse or not
        if reverse:
            idx_range.reverse()

        # Sort **self.coords**
        self.coords = self[:, idx_range]

        # Refill **self.atoms**
        symbols = self.symbol[idx_range]
        self.atoms = {}
        for i, at in enumerate(symbols):
            try:
                self.atoms[at].append(i)
            except KeyError:
                self.atoms[at] = [i]

        # Sort **self.bonds**
        if self.bonds is not None:
            self.atom1 = idx_range[self.atom1]
            self.atom2 = idx_range[self.atom2]
            self.bonds[:, 0:2].sort(axis=1)
            idx = self.bonds[:, 0:2].argsort(axis=0)[:, 0]
            self.bonds = self.bonds[idx]

    def residue_argsort(self, concatenate=True):
        """ Returns the indices that would sort **self** by residue number.
        Residues are defined based on moleculair fragments based on **self.bonds**.

        :parameter bool concatenate: If False, returned a nested list with atomic indices. Each
            sublist contains the indices of a single residue.
        :return: An array of indices that would sort *n* atoms **self**.
        :rtype: *n* |np.ndarray|_ [|np.int64|_].
        """
        # Define residues
        plams_mol = self.as_Molecule(mol_subset=0)[0]
        frags = plams_mol.separate_mod()

        # Sort the residues
        core = []
        ligands = []
        for frag in frags:
            if len(frag) == 1:
                core += frag
            else:
                ligands.append(frag)
        core.sort()
        ligands.sort()

        ret = [core] + ligands
        if concatenate:
            return np.concatenate(ret)
        return ret

    def get_center_of_mass(self, mol_subset=None, atom_subset=None):
        """ Get the center of mass.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all *m* molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all *n* atoms per molecule
            in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :return: An array with the centres of mass of *m* molecules with *n* atoms.
        :rtype: *m*3* |np.ndarray|_ [|np.float64|_].
        """
        coords = self.as_mass_weighted(mol_subset, atom_subset)
        return coords.sum(axis=1) / self.mass.sum()

    def get_bonds_per_atom(self, atom_subset=None):
        """ Get the number of bonds per atom in **self**.

        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all *n* atoms per molecule
            in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :return: An array with the number of bonds per atom, for all *n* atoms in **self**.
        :rtype: *n* |np.ndarray|_ [|np.int64|_].
        """
        j = self._get_atom_subset(atom_subset)
        if self.bonds is None:
            return np.zeros(len(j), dtype=int)
        return np.bincount(self.bonds[:, 0:2].flatten(), minlength=self.shape[1])[j]

    """ ################################## Root Mean Squared ################################## """

    def init_rmsd(self, mol_subset=None, atom_subset=None, reset_origin=True):
        """ Initialize the RMSD calculation, returning a dataframe.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all *m* molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all *n* atoms per molecule
            in **self** if *None*.
        :type atom_subset:  |None|_, |int|_ or |str|_
        :parameter bool reset_origin: Reset the origin of each molecule in **self** by means of
            a partial Procrustes superimposition, translating and rotating the molecules.
        :return: A dataframe of RMSDs with one column for every string or list of ints in
            **atom_subset**. Keys consist of atomic symbols (*e.g.* 'Cd') if **atom_subset**
            contains strings, otherwise a more generic 'series ' + str(int) scheme is adopted
            (*e.g.* 'series 2'). Molecular indices are used as indices.
        :rtype: |pd.DataFrame|_ (keys: |str|_, values: |np.float64|_, indices: |np.int64|_).
        """
        # Set the origin of all frames to their center of mass
        if reset_origin:
            self.reset_origin(mol_subset, inplace=True)

        # Figure out if the RMSD should be calculated as a single or multiple series
        atom_subset = atom_subset or tuple(self.atoms.keys())
        loop = self._get_loop(atom_subset)

        # Get the RMSD
        if loop:
            rmsd = np.array([self.get_rmsd(mol_subset, at) for at in atom_subset]).T
        else:
            rmsd = self.get_rmsd(mol_subset, atom_subset)

        # Construct arguments for the dataframe
        columns = self._get_rmsd_columns(rmsd, loop, atom_subset)
        index = mol_subset or np.arange(0, self.shape[0])
        data = rmsd

        # Create, fill and return a dataframe with the RMSD
        df = pd.DataFrame(data=data, index=index, columns=columns)
        df.columns.name = 'RMSD  /  Ångström'
        df.index.name = 'XYZ frame number'
        return df

    def init_rmsf(self, mol_subset=None, atom_subset=None, reset_origin=True):
        """ Initialize the RMSF calculation, returning a dataframe.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all *m* molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all *n* atoms per molecule
            in **self** if *None*.
        :type atom_subset:  |None|_, |int|_ or |str|_
        :parameter bool reset_origin: Reset the origin of each molecule in **self** by means of
            a partial Procrustes superimposition, translating and rotating the molecules.
        :return: A dataframe of RMSFs with one column for every string or list of ints in
            **atom_subset**. Keys consist of atomic symbols (*e.g.* 'Cd') if **atom_subset**
            contains strings, otherwise a more generic 'series ' + str(int) scheme is adopted
            (*e.g.* 'series 2'). Molecular indices are used as indices.
        :rtype: |pd.DataFrame|_ (keys: |str|_, values: |np.float64|_, indices: |np.int64|_).
        """
        # Set the origin of all frames to their center of mass
        if reset_origin:
            self.reset_origin(mol_subset, inplace=True)

        # Figure out if the RMSD should be calculated as a single or multiple series
        atom_subset = atom_subset or tuple(self.atoms.keys())
        loop = self._get_loop(atom_subset)

        # Get the RMSF
        if loop:
            rmsf = [self.get_rmsf(mol_subset, at) for at in atom_subset]
        else:
            rmsf = self.get_rmsf(mol_subset, atom_subset)

        # Construct arguments for the dataframe
        index = np.arange(0, self.shape[1])
        columns, data = self._get_rmsf_columns(rmsf, index, loop=loop, atom_subset=atom_subset)

        # Create, fill and return a dataframe with the RMSF
        df = pd.DataFrame(data=data, index=index, columns=columns)
        df.columns.name = 'RMSF  /  Ångström'
        df.index.name = 'Arbitrary atomic index'
        return df

    def get_rmsd(self, mol_subset=None, atom_subset=None):
        """ Calculate the root mean square displacement (RMSD) with respect to the first molecule
        **self.coords**. Returns a dataframe with the RMSD as a function of the XYZ frame numbers.
        """
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Calculate and return the RMSD per molecule in **self.coords**
        dist = np.linalg.norm(self[i, j, :] - self[0, j, :], axis=2)
        return np.sqrt(np.einsum('ij,ij->i', dist, dist) / dist.shape[1])

    def get_rmsf(self, mol_subset=None, atom_subset=None):
        """ Calculate the root mean square fluctuation (RMSF) of **self.coords**.
        Returns a dataframe as a function of atomic indices. """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Calculate the RMSF per molecule in **self.coords**
        mean_coords = np.mean(self[i, j, :], axis=0)[None, ...]
        displacement = np.linalg.norm(self[i, j, :] - mean_coords, axis=2)**2
        return np.mean(displacement, axis=0)

    def _get_rmsd_columns(self, rmsd, loop=False, atom_subset=None):
        """ Return the columns for the RMSD dataframe. """
        if loop:  # Plan A: **atom_subset** is a *list* of *str* or nested *list* of *int*
            if isinstance(atom_subset[0], str):  # Use atomic symbols or general indices as keys
                columns = atom_subset
            else:
                columns = np.arange(len(atom_subset))
        else:  # Plan B: **atom_subset** is something else
            if isinstance(atom_subset, str):  # Use an atomic symbol or a general index as keys
                columns = [atom_subset]
            else:
                columns = [0]

        return columns

    def _get_rmsf_columns(self, rmsf, index, loop=False, atom_subset=None):
        """ Return the columns and data for the RMSF dataframe. """
        if loop:  # Plan A: **atom_subset** is a *list* of *str* or nested *list* of *int*
            if isinstance(atom_subset[0], str):  # Use atomic symbols or general indices as keys
                columns = atom_subset
            else:
                columns = np.arange(len(atom_subset))

            # Create and fill a padded array
            data = np.full((len(rmsf), index.shape[0]), np.nan)
            k = 0
            for i, j in enumerate(rmsf):
                data[i, k:k+len(j)] = j
                k += len(j)
            data = data.T

        else:  # Plan B: **atom_subset** is something else
            if isinstance(atom_subset, str):  # Use an atomic symbol or a general index as keys
                columns = [atom_subset]
            else:
                columns = [0]

            # Create and fill a padded array
            data = np.full((index.shape[0]), np.nan)
            idx = self._get_atom_subset(atom_subset)
            data[idx] = rmsf

        return columns, data

    def _get_loop(self, subset):
        """ Figure out if the supplied subset warrants a for loop or not. """
        if subset is None:
            return True  # subset is *None*
        elif isinstance(subset, np.ndarray):
            subset = subset.tolist()

        if isinstance(subset, str):
            return False  # subset is a *str*
        elif isinstance(subset, int):
            return False  # subset is an *int*
        elif isinstance(subset[0], str):
            return True  # subset is an iterable of *str*
        elif isinstance(subset[0], int):
            return False  # subset is an iterable of *str*
        elif isinstance(subset[0][0], int):
            return True  # subset is a nested iterable of *str*
        raise TypeError()

    """ #############################  Radial Distribution Functions  ######################### """

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
        mol_subset = self._get_mol_subset(mol_subset)
        i = mol_subset, self._get_atom_subset(atom_subset[0])
        j = mol_subset, self._get_atom_subset(atom_subset[1])

        # Slice the XYZ array
        A = self[i]
        B = self[j]

        # Create, fill and return the distance matrix
        if A.ndim == 2:
            return cdist(A, B)[None, ...]
        shape = A.shape[0], A.shape[1], B.shape[1]
        ret = np.empty(shape)
        for k, (a, b) in enumerate(zip(A, B)):
            ret[k] = cdist(a, b)
        return ret

    """ ############################  Angular Distribution Functions  ######################### """

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
        mol_subset = self._get_mol_subset(mol_subset)
        i = mol_subset, self._get_atom_subset(atom_subset[0])
        j = mol_subset, self._get_atom_subset(atom_subset[1])
        k = mol_subset, self._get_atom_subset(atom_subset[2])

        # Slice and broadcast the XYZ array
        A = self[i][:, None, ...]
        B = self[j][..., None, :]
        C = self[k][..., None, :]

        # Temporary ignore RuntimeWarnings related to dividing by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # Prepare the unit vectors
            kwarg1 = {'atom_subset': [atom_subset[0], atom_subset[1]], 'mol_subset': mol_subset}
            kwarg2 = {'atom_subset': [atom_subset[0], atom_subset[2]], 'mol_subset': mol_subset}
            unit_vec1 = (A - B) / self.get_dist_mat(**kwarg1)[..., None]
            unit_vec2 = (A - C) / self.get_dist_mat(**kwarg2)[..., None]

            # Create and return the angle matrix
            return np.arccos(np.einsum('ijkl,ijml->ijkm', unit_vec1, unit_vec2))

    def _get_atom_subset(self, arg):
        """ Grab and return a list of indices from **self.atoms**.
        Return *at* if it is *None*, an *int* or iterable container consisting of *int*. """
        if arg is None:
            return slice(0, None)
        elif isinstance(arg, (int, np.integer)):
            return [arg]
        elif isinstance(arg[0], (int, np.integer)):
            return arg
        elif isinstance(arg, str):
            return self.atoms[arg]
        elif isinstance(arg[0], str):
            return list(chain.from_iterable(self.atoms[i] for i in arg))
        elif isinstance(arg[0][0], (int, np.integer)):
            return list(chain.from_iterable(arg))
        raise TypeError(str(type(arg)) + ': ' + str(arg) + ' is not a valid object type for /'
                        'the atom_subset argument')

    def _get_mol_subset(self, arg):
        """ """
        if arg is None:
            return slice(0, None)
        elif isinstance(arg, (int, np.integer)):
            return [arg]
        else:
            return arg

    """ #################################  Type conversion  ################################### """

    def _set_psf_block(self, inplace=True):
        """ """
        res = self.residue_argsort(concatenate=False)
        plams_mol = self.as_Molecule(0)[0]
        plams_mol.fix_bond_orders()

        df = pd.DataFrame(index=np.arange(1, self.shape[1]+1))
        df.index.name = 'ID'
        df['segment name'] = 'MOL'
        df['residue ID'] = [i for i, j in enumerate(res, 1) for _ in j]
        df['residue name'] = ['COR' if i == 1 else 'LIG' for i in df['residue ID']]
        df['atom name'] = self.symbol
        df['atom type'] = df['atom name']
        df['charge'] = [at.properties.charge for at in plams_mol]
        df['mass'] = self.mass
        df[0] = 0

        if not inplace:
            return df
        self.properties.atoms = df

    def _update_atom_type(self, filename='mol.param'):
        """ """
        if self.properties.atoms is None:
            self._set_psf_block()

        at_type = read_param(filename)['nonbonded'][0].values
        id_range = range(1, max(self.properties.atoms['residue ID']))
        for i in id_range:
            self.properties.atoms.loc[i, 'atom type'] = at_type

    def as_psf(self, filename='mol.psf'):
        """ Convert a *MultiMolecule* object into a Protein Structure File (.psf). """
        # Prepare the !NTITLE block
        top = 'PSF EXT\n'
        top += '\n' + '{:>8.8}'.format(str(2)) + ' !NTITLE'
        top += '\n' + '{:>8.8}'.format('REMARKS') + ' PSF file generated with Auto-FOX:'
        top += '\n' + '{:>8.8}'.format('REMARKS') + ' https://github.com/nlesc-nano/auto-FOX'
        top += '\n\n' + '{:>8.8}'.format(str(self.shape[1])) + ' !NATOM\n'

        # Prepare the !NATOM block
        if self.properties.atoms is None:
            self._set_psf_block()
        df = self.properties.atoms.T
        df.reset_index(level=0, inplace=True)
        mid = ''
        for key in df:
            string = '{:>8.8} {:4.4} {:4.4} {:4.4} {:4.4} {:5.5} {:>9.9} {:>13.13} {:>11.11}'
            mid += string.format(*[str(key)]+[str(i) for i in df[key]]) + '\n'

        # Prepare the !NBOND, !NTHETA, !NPHI and !NIMPHI blocks
        bottom = ''
        bottom_headers = ['{:>8.8} !NBOND: bonds', '{:>8.8} !NTHETA: angles',
                          '{:>8.8} !NPHI: dihedrals', '{:>8.8} !NIMPHI: impropers']
        if self.bonds is None:
            for header in bottom_headers:
                bottom += '\n\n' + header.format('0')
            bottom += '\n\n'
        else:
            plams_mol = self.as_Molecule(0)[0]
            plams_mol.fix_bond_orders()
            plams_mol.set_atoms_id(start=0)
            connectivity = [self.bonds[:, 0:2], plams_mol.get_angles(),
                            plams_mol.get_dihedrals(), plams_mol.get_impropers()]
            items_per_row = [4, 3, 2, 2]

            for i, conn, header in zip(items_per_row, connectivity, bottom_headers):
                bottom += '\n\n' + header.format(str(len(conn)))
                bottom += '\n' + serialize_array(conn, i)
            bottom += '\n'

        # Export the .psf file
        with open(filename, 'w') as file:
            file.write(top)
            file.write(mid)
            file.write(bottom[1:])

    def _mol_to_file(self, filename, outputformat, mol_subset=0):
        """ Create files using the plams.Molecule.write() method. """
        mol_list = self.as_Molecule(mol_subset)

        if len(mol_list) != 1:
            name_list = filename.rsplit('.', 1)
            name_list.insert(-1, '.{}.')
            name = ''.join(name_list)
        else:
            name = filename

        for i, plams_mol in enumerate(mol_list, 1):
            plams_mol.write(name.format(str(i)), outputformat=outputformat)

    def as_pdb(self, mol_subset=0, filename='mol.pdb'):
        """ Convert a *MultiMolecule* object into one or more Protein DataBank files (.pdb). """
        self._mol_to_file(filename, 'pdb', mol_subset)

    def as_mol2(self, mol_subset=0, filename='mol.mol2'):
        """ Convert a *MultiMolecule* object into one or more .mol2 files. """
        self._mol_to_file(filename, 'mol2', mol_subset)

    def as_mol(self, mol_subset=0, filename='mol.mol'):
        """ Convert a *MultiMolecule* object into one or more .mol files. """
        self._mol_to_file(filename, 'mol', mol_subset)

    def as_xyz(self, filename='mol.xyz'):
        """ Convert a *MultiMolecule* object into an .xyz file.

        :parameter str filename: The path+filename (including extension) of the
            to be created .xyz file.
        """
        # Define constants
        at = self.symbol[:, None]
        header = str(len(at)) + '\n' + 'frame '
        kwarg = {'fmt': ['%-2.2s', '%-10.10s', '%-10.10s', '%-10.10s'],
                 'delimiter': '     ', 'comments': ''}

        # Create the .xyz file
        with open(filename, 'wb') as file:
            for i, xyz in enumerate(self, 1):
                np.savetxt(file, np.hstack((at, xyz)), header=header+str(i), **kwarg)

    def as_mass_weighted(self, mol_subset=None, atom_subset=None, inplace=False):
        """ Transform the Cartesian of **self.coords** into mass-weighted Cartesian coordinates.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all *m* molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all *n* atoms per molecule
            in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :parameter bool inplace: Instead of returning the new coordinates, perform an inplace
            update of **self.coords**.
        :return: An array of mass-weighted Cartesian coordinates of *m* molecules with *n* atoms
            and, optionally, an array of *n* atomic masses.
        :rtype: *m*n*3* |np.ndarray|_ [|np.float64|_] and, optionally,
            *n* |np.ndarray|_ [|np.float64|_]
        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Create an array of mass-weighted Cartesian coordinates
        if inplace:
            self.coords = self[i, j, :] * self.mass[None, j, None]
        else:
            return self[i, j, :] * self.mass[None, j, None]

    def from_mass_weighted(self, mol_subset=None, atom_subset=None):
        """ Transform **self.coords** from mass-weighted Cartesian into Cartesian coordinates.
        Performs an inplace update of **self.coords**.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all *m* molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all *n* atoms per molecule
            in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Update **self.coords**
        self[i, j, :] /= self.mass[None, j, None]

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
        mol_subset = self._get_mol_subset(mol_subset)
        if atom_subset is None:
            atom_subset = np.arange(self.shape[1])
        else:
            atom_subset = sorted(self._get_atom_subset(atom_subset))
        at_symbols = self.symbol

        # Construct a template molecule and fill it with atoms
        assert self.coords is not None
        assert self.atoms is not None
        mol_template = Molecule()
        mol_template.properties = self.properties.copy()
        for i in atom_subset:
            atom = Atom(symbol=at_symbols[i])
            mol_template.add_atom(atom)

        # Fill the template molecule with bonds
        if self.bonds is not None:
            bond_idx = np.ones(len(self))
            bond_idx[atom_subset] += np.arange(len(atom_subset))
            for i, j, order in self.bonds:
                if i in atom_subset and j in atom_subset:
                    at1 = mol_template[int(bond_idx[i])]
                    at2 = mol_template[int(bond_idx[j])]
                    mol_template.add_bond(Bond(atom1=at1, atom2=at2, order=order/10.0))

        # Create copies of the template molecule; update their cartesian coordinates
        ret = []
        for i, xyz in enumerate(self[mol_subset]):
            mol = mol_template.copy()
            mol.from_array(xyz[atom_subset])
            mol.properties.frame = i
            ret.append(mol)

        return ret

    def from_Molecule(self, mol_list, subset=None, allow_different_length=False):
        """ Convert a list of PLAMS molecules into a *MultiMolecule* object.
        Performs an inplace modification of **self**.

        :parameter mol_list: A PLAMS molecule or list of PLAMS molecules.
        :type mol_list: |plams.Molecule|_ or |list|_ [|plams.Molecule|_]
        :parameter subset: Transfer a subset of *plams.Molecule* attributes to **self**. If *None*,
            transfer all attributes. Accepts one or more of the following values as strings:
            *properties*, *atoms* and/or *bonds*.
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        :parameter bool allow_different_length: If *True*, allow **mol_list** and **self.coords**
            to have different lengths.
        """
        if isinstance(mol_list, Molecule):
            plams_mol = mol_list
        else:
            plams_mol = mol_list[0]
        subset = subset or ('atoms', 'bonds', 'properties')

        if not allow_different_length:
            # Raise an error if mol_list and self.coords are of different lengths
            if len(mol_list) != self.shape[0]:
                error = 'from_Molecule: Shape mismatch, the mol_list is of length '
                error += str(len(mol_list)) + ' while self.coords is of length: '
                error += str(self.shape[0])
                raise IndexError(error)

        # Convert properties
        if 'properties' in subset:
            self.properties = plams_mol.properties

        # Convert atoms
        if 'atoms' in subset:
            dummy = Molecule()
            idx = slice(0, None)
            self.coords = np.array([dummy.as_array(atom_subset=mol.atoms[idx]) for mol in mol_list])
            for i, at in enumerate(plams_mol.atoms[idx]):
                try:
                    self.atoms[at].append(i)
                except KeyError:
                    self.atoms[at] = [i]

        # Convert bonds
        if 'bonds' in subset:
            plams_mol.set_atoms_id()
            self.bonds = np.empty((len(plams_mol.bonds), 3), dtype=int)
            for i, bond in enumerate(plams_mol.bonds):
                self.bonds[i] = bond.atom1.id, bond.atom2.id, bond.order * 10
            self.bonds[:, 0:2] -= 1
            plams_mol.unset_atoms_id()

    """ ####################################  Copying  ######################################## """

    def copy(self, subset=None, deep=False):
        """ Create and return a new *MultiMolecule* object and fill its attributes with
        views of their respective counterparts in **self**, creating a shallow copy.

        :parameter subset: Copy a subset of attributes from **self**; if *None*, copy all
            attributes. Accepts one or more of the following attribute names as strings: *coords*,
            *atoms*, *bonds* and/or *properties*.
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        :parameter bool deep: If *True*, perform a deep copy instead of a shallow copy.
        """
        # Perform a deep copy instead of a shallow copy
        if deep:
            return self.deepcopy(subset)

        attr_dict = vars(self)
        subset = subset or attr_dict
        if isinstance(subset, str):
            subset = (subset)

        ret = MultiMolecule()
        for i in attr_dict:
            if i in subset:
                setattr(ret, i, attr_dict[i])
        return ret

    def deepcopy(self, subset=None):
        """ Create and return a new *MultiMolecule* object and fill its attributes with
        copies of their respective counterparts in **self**, creating a deep copy.

        :parameter subset: Deep copy a subset of attributes from **self**; if *None*, deep copy all
            attributes. Accepts one or more of the following attribute names as strings: *coords*,
            *atoms*, *bonds* and/or *properties*.
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        """
        attr_dict = vars(self)
        subset = subset or attr_dict
        if isinstance(subset, str):
            subset = (subset)

        ret = MultiMolecule()
        for i in attr_dict:
            if i in subset:
                try:
                    setattr(ret, i, attr_dict[i].copy())
                except AttributeError:
                    pass
        return ret
