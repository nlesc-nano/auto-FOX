""" A Module for the MultiMolecule class. """

__all__ = ['MultiMolecule']

from itertools import chain

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scm.plams import (Atom, Bond, Molecule)
from scm.plams import PeriodicTable

from .rdf import (get_rdf, get_rdf_lowmem, get_rdf_df)
from .multi_mol_magic import _MultiMolecule


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
    :type atoms: |None|_ or |plams.Settings|_ (keys: |str|_, values: |list|_ [|int|_],
                                               superclass: |dict|_)
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

    def guess_bonds(self, atom_subset=None):
        """ Guess bonds within the molecules based on atom type and inter-atomic distances.
        Bonds are guessed based on the first molecule in **self.coords**
        Performs an inplace modification of **self.bonds**

        :parameter atom_subset: A tuple of atomic symbols. Bonds are guessed between all atoms
            whose atomic symbol is in **atom_subset**. If *None*, guess bonds for all atoms in
            **self.coords**.
        :type atom_subset: |None|_ or |tuple|_ [|str|_]
        """
        atom_subset = self._subset_to_idx(atom_subset)
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

    def reset_origin(self, mol_subset=None, atom_subset=None, inplace=True):
        """ Set the origin to the center of mass of **self.coords**. Performs in inplace update
        of **self.coords** if **inplace** is *True*.

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
        i = mol_subset or slice(0, self.shape[0])
        j = self._subset_to_idx(atom_subset) or slice(0, self.shape[1])

        # Remove translations
        coords = self[i, j, :] - np.average(self[i, j, :], axis=1)[:, None, :]

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

    def get_atomic_property(self, prop='symbol'):
        """ Take **self.atoms** and return an (concatenated) array of a specific property associated
        with an atom type. Values are sorted by their indices.

        :parameter str prop: The to be returned property. Accepted values:
            **symbol**, **atnum**, **mass**, **radius** or **connectors**.
            See the |PeriodicTable|_ module of PLAMS for more details.
        :return: A dictionary with atomic indices as keys and atomic symbols as values.
        :rtype: |np.array|_ [|np.float64|_, |str|_ or |np.int64|_].
        """
        def get_symbol(symbol):
            """ Takes an atomic symbol and returns itself. """
            return symbol

        # Interpret the **values** argument
        prop_dict = {
                'symbol': get_symbol,
                'radius': PeriodicTable.get_radius,
                'atnum': PeriodicTable.get_atomic_number,
                'mass': PeriodicTable.get_mass,
                'connectors': PeriodicTable.get_connectors
        }

        # Create concatenated lists of the keys and values in **self.atoms**
        value_list = list(chain.from_iterable(self.atoms.values()))
        key_list = []
        for at in self.atoms:
            key = prop_dict[prop](at)
            key_list += [key for _ in self.atoms[at]]

        # Sort and return
        return np.array([key for _, key in sorted(zip(value_list, key_list))])

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
        mass = self.get_atomic_property(prop='mass')
        return coords.sum(axis=1) / mass.sum()

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
        i = mol_subset or slice(0, self.shape[0])
        j = self._subset_to_idx(atom_subset) or slice(0, self.shape[1])

        # Calculate and return the RMSD per molecule in **self.coords**
        dist = np.linalg.norm(self[i, j, :] - self[0, j, :], axis=2)
        return np.sqrt(np.einsum('ij,ij->i', dist, dist) / dist.shape[1])

    def get_rmsf(self, mol_subset=None, atom_subset=None):
        """ Calculate the root mean square fluctuation (RMSF) of **self.coords**.
        Returns a dataframe as a function of atomic indices. """
        # Prepare slices
        i = mol_subset or slice(0, self.shape[0])
        j = self._subset_to_idx(atom_subset) or slice(0, self.shape[1])

        # Calculate the RMSF per molecule in **self.coords**
        mean_coords = np.average(self[i, j, :], axis=0)[None, :, :]
        displacement = np.linalg.norm(self[i, j, :] - mean_coords, axis=2)**2
        return np.average(displacement, axis=0)

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
            idx = self._subset_to_idx(atom_subset)
            data[idx] = rmsf

        return columns, data

    def _get_loop(self, subset):
        """ Figure out if the supplied subset warrants a for loop or not. """
        if subset is None:
            return True  # subset is *None*
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
        raise TypeError('')

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
        mol_subset = mol_subset or slice(0, self.shape[0])
        i = mol_subset, self._subset_to_idx(atom_subset[0]) or slice(0, self.shape[1])
        j = mol_subset, self._subset_to_idx(atom_subset[1]) or slice(0, self.shape[1])

        # Slice the XYZ array
        A = self[i]
        B = self[j]

        # Create, fill and return the distance matrix
        shape = self.shape[0], A.shape[1], B.shape[1]
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
        mol_subset = mol_subset or slice(0, self.shape[0])
        i = mol_subset, self._subset_to_idx(atom_subset[0]) or slice(0, self.shape[1])
        j = mol_subset, self._subset_to_idx(atom_subset[1]) or slice(0, self.shape[1])
        k = mol_subset, self._subset_to_idx(atom_subset[2]) or slice(0, self.shape[1])

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

    def _subset_to_idx(self, arg):
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
        elif isinstance(arg[0][0], int):
            return list(chain.from_iterable(arg))
        raise KeyError()

    """ #################################  Type conversion  ################################### """

    def as_mass_weighted(self, mol_subset=None, atom_subset=None, inplace=False):
        """ Transform Cartesian into mass-weighted Cartesian coordinates.

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
        # Prepare an array of atomic masses
        mass = self.get_atomic_property(prop='mass')

        # Prepare slices
        i = mol_subset or slice(0, self.shape[0])
        j = self._subset_to_idx(atom_subset) or slice(0, self.shape[1])

        # Create an array of mass-weighted Cartesian coordinates
        mass_weighted = self[i, j, :] * mass[None, j, None]

        if inplace:
            self.coords = mass_weighted
        else:
            return mass_weighted

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
        # Prepare an array of atomic masses
        mass = self.get_atomic_property(prop='mass')

        # Prepare slices
        i = mol_subset or slice(0, self.shape[0])
        j = self._subset_to_idx(atom_subset) or slice(0, self.shape[1])

        # Update **self.coords**
        self[i, j, :] /= mass[None, j, None]

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
        atom_subset = self._subset_to_idx(atom_subset)
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
        :type mol_list: |plams.Molecule|_ or |list|_ [|plams.Molecule|_]
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

    """ ####################################  Copying  ######################################## """

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
