from itertools import chain

import numpy as np
from scipy.spatial.distance import cdist

from scm.plams.mol.molecule import Molecule
from scm.plams.mol.atom import Atom
from scm.plams.mol.bond import Bond

from FOX.functions.multi_mol_magic import _MultiMolecule
from FOX.functions.rdf import (get_rdf, get_rdf_lowmem, get_rdf_df)


class MultiMolecule(_MultiMolecule):
    def guess_bonds(self, atoms=None):
        """ Guess bonds within the molecule(s) based on atom type and inter-atomic distances.
        It is assumed that connectivity is identical for all molecules.

        :parameter atoms: A tuple of atomic symbols. Bonds are guessed for all atoms with matching
            atomic symbols.
        :type atoms: None or |tuple|_ [|str|_]
        """
        idx_list = list(chain.from_iterable([self[at] for at in atoms]))
        mol = self.as_Molecule(mol_subset=0, atom_subset=idx_list)
        mol.guess_bonds()
        self.from_Molecule(mol, subset='bonds')

    def remove_random_coords(self, p=0.5):
        """ Remove random xyz frames from **self.coords**.
        The probability of removing a specific frame is equal to **p**.

        :parameter float p: The probability of remove a 2D slice from **self.coords**.
            Accepts values between 0.0 (0%) and 1.0 (100%).
        """
        if p <= 0.0 or p >= 1.0:
            raise IndexError('The probability, p, must be larger than 0.0 and smaller than 1.0')
        elif self.shape[0] == 1:
            raise IndexError('Grabbing random 2D slices from a 2D array makes no sense')

        size = 1 or int(self.shape[0] / p)
        idx_range = np.arange(self.shape[0])
        idx = np.random.choice(idx_range, size)
        self.coords = self[idx]

    def get_angle_mat(self, frame=0, atoms=[None, None, None]):
        """ Create and return an angle matrix. """
        # Define array slices
        frame = frame or slice(0, self.shape[0])
        i = frame, self.get_idx(atoms[0]) or slice(0, self.shape[1])
        j = frame, self.get_idx(atoms[1]) or slice(0, self.shape[1])
        k = frame, self.get_idx(atoms[2]) or slice(0, self.shape[1])

        # Slice and broadcast the XYZ array
        A = self[i][:, None, :]
        B = self[j][:, :, None]
        C = self[k][:, :, None]

        # Prepare the unit vectors
        kwarg1 = {'atoms': [atoms[0], atoms[1]], 'frame': frame}
        kwarg2 = {'atoms': [atoms[0], atoms[2]], 'frame': frame}
        unit_vec1 = A - B / self.get_dist_mat(**kwarg1)
        unit_vec2 = A - C / self.get_dist_mat(**kwarg2)

        # Create and return the angle matrix
        ret = np.arccos(np.einsum('ijkl,ijml->ijkl', unit_vec1, unit_vec2))
        if ret.shape[0] == 1:
            return ret[0]
        return ret

    #############################################  rdf  ###########################################

    def init_rdf(self, atoms=None, dr=0.05, r_max=12.0, low_mem=False):
        """ Calculate and return the radial distribution functions (RDFs) for all possible atom-pairs
        in **atoms** as a dataframe. Accepts both 2d and 3d arrays of cartesian coordinates as input.

        :parameter atoms: A tuple of atomic symbols. RDFs will be calculated for all
            possible atom-pairs in **atoms**. If *None*, calculate RDFs for all possible atom-pairs
            in the keys of **idx_dict** (*i.e.* all possible atoms pairs in the molecule).
        :type atoms: None or tuple [str]
        :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
            concentric spheres.
        :parameter float r_max: The maximum to be evaluated interatomic distance.
        :parameter float low_mem: If *True*, use a slower but more memory efficient method for
            constructing the RDFs.
        :return: A dataframe of radial distribution functions, averaged over all conformations in
            **xyz_array**. Keys are of the form: at_symbol1 + ' ' + at_symbol2 (*e.g.* 'Cd Cd').
            The radius is used as index.
        :rtype: |pd.DataFrame|_ (keys: |str|_, values: |np.float64|_, indices: |np.float64|_).
        """
        # If *atoms* is None: extract atomic symbols from they keys of *idx_dict*
        atoms = atoms or tuple(self.atoms.keys())

        # Construct a list of 2-tuples containing all unique atom pairs
        atom_pairs = [(at1, at2) for i, at1 in enumerate(atoms) for at2 in atoms[i:]]

        # Construct an empty dataframe with appropiate dimensions, indices and keys
        df = get_rdf_df(dr, r_max, atom_pairs)

        # Fill the dataframe with RDF's, summed over all conformations in mol_list
        kwarg1 = {'dr': dr, 'r_max': r_max}
        if low_mem:
            # Slower low mem approach
            kwarg2 = {'frame': None, 'atoms': None}
            for i, _ in enumerate(self):
                kwarg2['frame'] = i
                for at1, at2 in atom_pairs:
                    kwarg2['atoms'] = (at1, at2)
                    df[at1 + ' ' + at2] += get_rdf_lowmem(self.get_dist_mat(**kwarg2), **kwarg1)

            df.iloc[0] = 0.0
            df /= self.shape[0]
        else:
            # Faster high mem approach
            kwarg2 = {'frame': None, 'atoms': None}
            for at1, at2 in atom_pairs:
                kwarg2['atoms'] = (at1, at2)
                df[at1 + ' ' + at2] = get_rdf(self.get_dist_mat(**kwarg2), **kwarg1)

        return df

    def get_dist_mat(self, frame=None, atoms=[None, None]):
        """ Create and return a distance matrix.
        Return a 3D or 2D distance matrix. """
        # Define array slices
        frame = frame or slice(0, self.shape[0])
        i = frame, self.get_idx(atoms[0]) or slice(0, self.shape[1])
        j = frame, self.get_idx(atoms[1]) or slice(0, self.shape[1])

        # Slice the XYZ array
        A = self[i]
        B = self[j]

        # Create, fill and return the distance matrix
        shape = self.shape[0], A.shape[1], B.shape[1]
        ret = np.empty(shape)
        for k, (a, b) in enumerate(zip(A, B)):
            ret[k] = cdist(a, b)
        if ret.shape[0] == 1:
            return ret[0]
        return ret

    def get_idx(self, at):
        """ Grab and return a list of indices from **self.atoms**.
        Retun *None* if **at** is *None*. """
        if at is None:
            return at
        else:
            return self.atoms[at]

    #######################################  Type conversion  #####################################

    def as_Molecule(self, mol_subset=None, atom_subset=None):
        """ Convert a *MultiMolecule* object into a *list* of *plams.Molecule*.

        :parameter mol_subset: Convert a subset of molecules in **self.coords** based on their
            indices. If *None*, convert all molecules.
        :type mol_subset: |None|_, |int|_ or |tuple|_ [|int|_]
        :parameter atom_subset: Convert a only subset of atoms within each molecule in
            **self.coords**, as based on their indices. If *None*, convert all atoms per molecule.
        :type atom_subset: |None|_, |int|_ or |tuple|_ [|int|_]
        :return: A PLAMS molecule or list of PLAMS molecules.
        :rtype: |plams.Molecule|_ or |list|_ [|plams.Molecule|_].
        """
        # Create a dictionary with atomic indices as keys and matching atomic symbols as values
        idx_dict = self._invert_atoms()

        # Construct a template molecule and fill it with atoms
        mol_template = Molecule()
        for i in idx_dict:
            at = Atom(symbol=idx_dict[i])
            mol_template.add_atom(at)

        # Fill the template molecule with bonds
        for bond, order in zip(self.bonds, self.bond_orders):
            i = int(bond[0]) + 1
            j = int(bond[1]) + 1
            bond = Bond(atom1=mol_template[i], atom2=mol_template[j], order=order)
            mol_template.add_bond(bond)

        # Create copies of the template molecule; update their cartesian coordinates
        ret = []
        idx_range = mol_subset or np.arange(1, self.shape[0])
        for i, xyz in zip(self, idx_range):
            mol = mol_template.copy()
            mol.from_array(xyz)
            mol.properties.frame = i
            ret.append(mol)

        # Return a molecule or list of molecules
        if ret.shape[0] == 1:
            return ret[0]
        return ret

    def from_Molecule(self, mol_list, subset=None):
        """ Convert a *list* of *plams.Molecule* into a *MultiMolecule* object.

        :parameter mol_list: A list of PLAMS molecules.
        :type mol_list: |list|_ [|plams.Molecule|_].
        :parameter subset: Transfer a subset of *plams.Molecule* attributes to **self**. If *None*,
            transfer all attributes (*i.e.* 'properties', 'atoms' and 'bonds').
        :type subset: |None|_, |str|_ or |tuple|_ [|str|_]
        """
        mol = mol_list[0]
        subset = subset or ('atoms', 'bonds', 'properties')

        # Convert properties
        if 'properties' in subset:
            self.properties = mol.properties

        # Convert atoms
        if 'atoms' in subset:
            self.coords = np.array([mol.as_array() for mol in mol_list])
            self.shape = self.coords.shape
            self.dtype = self.coords.dtype
            for i, at in enumerate(mol):
                try:
                    self.atoms[at].append(i)
                except KeyError:
                    self.atoms[at] = [i]

        # Convert bonds
        if 'bonds' in subset:
            mol.set_atoms_id()
            self.bonds = np.empty((len(mol.bonds), 2), dtype=int)
            self.bond_orders = np.empty((len(mol.bonds)), dtype=float)
            for i, bond in enumerate(mol.bonds):
                self.bonds[i] = bond.at1.id, bond.at2.id
                self.bond_orders[i] = bond.order
            mol.unset_atoms_id()

    def _invert_atoms(self):
        """ Invert the **self.atoms** attribute, turing keys into values and values into keys. """
        ret = {}
        for at in self.atoms:
            for i in self.atoms[at]:
                ret[i] = at
        return ret

    #########################################  Copy  ##############################################

    def __copy__(self):
        """ Magic method, create and return a new *MultiMolecule* and fill its attributes with
        views of **self**. """
        ret = MultiMolecule()
        ret.coords = self.coords
        ret.shape = self.shape
        ret.dtype = self.dtype
        ret.atoms = self.atoms
        ret.bonds = self.bonds
        ret.bond_orders = self.bond_orders
        ret.properties = self.properties
        return ret

    def __deepcopy__(self):
        """ Magic method, create and return a deep copy of **self**. """
        ret = self.copy()
        ret.coords = self.coords.copy()
        ret.atoms = self.atoms.copy()
        ret.bonds = self.bonds.copy()
        ret.bond_orders = self.bond_orders.copy()
        ret.properties = self.properties.copy()
        return ret
