"""A Module for the MultiMolecule class."""

from __future__ import annotations

from itertools import (chain, combinations_with_replacement)
from typing import (
    Container, Optional, Union, List, Hashable, Callable, Iterable, Dict, Tuple, Any
)

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from scm.plams import (Atom, Bond)

from .molecule_utils import Molecule
from .multi_mol_magic import _MultiMolecule
from ..io.read_kf import read_kf
from ..io.read_xyz import read_multi_xyz
from ..functions.rdf import (get_rdf, get_rdf_lowmem, get_rdf_df)
from ..functions.adf import (get_adf, get_adf_df)

__all__ = ['MultiMolecule']

MolSubset = Union[None, slice, int]
AtomSubset = Union[
    None, slice, int, str, Container[int], Container[str], Container[Container[int]]
]


class MultiMolecule(_MultiMolecule):
    """A class designed for handling a and manipulating large numbers of molecules.

    More
    specifically, different conformations of a single molecule as derived from, for example,
    an intrinsic reaction coordinate calculation (IRC) or a molecular dymanics trajectory (MD).
    The class has access to four attributes (further details are provided under parameters):

    :parameter coords: A 3D array with the cartesian coordinates of :math:`m` molecules with
        :math:`n` atoms.
    :type coords: :math:`m*n*3` |np.ndarray|_ [|np.float64|_]
    :parameter atoms: A dictionary with atomic symbols as keys and matching atomic
        indices as values. Stored in the **atoms** attribute.
    :type atoms: |None|_ or |dict|_ (keys: |str|_, values: |list|_ [|int|_])
    :parameter bonds: A 2D array with indices of the atoms defining all :math:`k` bonds
        (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
        Stored in the **bonds** attribute.
    :type bonds: |None|_ or :math:`k*3` |np.ndarray|_ [|np.int64|_]
    :parameter properties: A Settings object (subclass of dictionary) intended for storing
        miscellaneous user-defined (meta-)data. Is devoid of keys by default. Stored in the
        **properties** attribute.
    :type properties: |plams.Settings|_
    """

    def guess_bonds(self, atom_subset: AtomSubset = None) -> None:
        """Guess bonds within the molecules based on atom type and inter-atomic distances.

        Bonds are guessed based on the first molecule in **self**
        Performs an inplace modification of **self.bonds**

        :parameter atom_subset: A tuple of atomic symbols. Bonds are guessed between all atoms
            whose atomic symbol is in **atom_subset**. If *None*, guess bonds for all atoms in
            **self**.
        :type atom_subset: |None|_ or |tuple|_ [|str|_]
        """
        if atom_subset is None:
            at_subset = np.arange(0, self.shape[1])
        else:
            at_subset = np.array(sorted(self._get_atom_subset(atom_subset)))

        # Guess bonds
        mol = self.as_Molecule(mol_subset=0, atom_subset=atom_subset)[0]
        mol.guess_bonds()
        mol.fix_bond_orders()
        self.bonds = MultiMolecule.from_Molecule(mol, subset='bonds').bonds

        # Update indices in **self.bonds** to account for **atom_subset**
        self.atom1 = at_subset[self.atom1]
        self.atom2 = at_subset[self.atom2]
        self.bonds[:, 0:2].sort(axis=1)
        idx = self.bonds[:, 0:2].argsort(axis=0)[:, 0]
        self.bonds = self.bonds[idx]

    def slice(self, start: int = 0,
              stop: Union[int, None] = None,
              step: int = 1,
              inplace: bool = False) -> Optional[MultiMolecule]:
        """Construct a new *MultiMolecule* by iterating through **self**
        along a set interval.

        Equivalent to :code:`MultiMolecule[start:stop:step].copy()` or
        :code:`MultiMolecule[start:stop:step]` depending on the value of **inplace**.

        :parameter int start: Start of the interval.
        :parameter int stop: End of the interval.
        :parameter int step: Spacing between values.
        :parameter bool inplace: Instead of returning the new coordinates, perform an inplace
            update of **self**.
        """
        if inplace:
            self[:] = self[start:stop:step]
        else:
            return self[start:stop:step].copy()

    def random_slice(self, start: int = 0,
                     stop: Union[int, None] = None,
                     p: float = 0.5,
                     inplace: bool = False) -> Optional[MultiMolecule]:
        """Construct a new *MultiMolecule* by iterating through **self** at random
        intervals.

        The probability of including a particular element is equivalent to **p**.

        :parameter int start: Start of the interval.
        :parameter int stop: End of the interval.
        :parameter float p: The probability of including each particular molecule in
            **self**. Values must be between :math:`0.0` (0%) and :math:`1.0` (100%).
        :parameter bool inplace: Instead of returning the new coordinates, perform an inplace
            update of **self**.
        """
        if p <= 0.0 or p >= 1.0:
            raise ValueError("The supplied probability, 'p': {:f}, must be larger "
                             "than 0.0 and smaller than 1.0".format(p))

        stop = stop or self.shape[0]
        idx_range = np.arange(start, stop)
        size = int(p * len(idx_range))
        idx = np.random.choice(idx_range, size=size, replace=False)

        if inplace:
            self[:] = self[idx]
        else:
            return self[idx].copy()

    def reset_origin(self, mol_subset: MolSubset = None,
                     atom_subset: AtomSubset = None,
                     inplace: bool = True) -> Optional[MultiMolecule]:
        """Reallign all molecules in **self**, rotating and translating them, by performing a
        partial partial Procrustes superimposition.

        The superimposition is carried out with respect to the first molecule in **self**.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all :math:`m` molecules in **self** if
            *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :parameter bool inplace: Instead of returning the new coordinates, perform an inplace
            update of **self**.
        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Remove translations
        coords = self[i, j, :] - self[i, j, :].mean(axis=1)[:, None, :]

        # Peform a singular value decomposition on the covariance matrix
        H = np.swapaxes(coords[0:], 1, 2) @ coords[0]
        U, _, Vt = np.linalg.svd(H)
        V, Ut = np.swapaxes(Vt, 1, 2), np.swapaxes(U, 1, 2)

        # Construct the rotation matrix
        rotmat = np.ones_like(U)
        rotmat[:, 2, 2] = np.linalg.det(V @ Ut)
        rotmat *= V@Ut

        # Return or perform an inplace update of **self**
        if inplace:
            self[i, j, :] = coords @ np.swapaxes(rotmat, 1, 2)
        else:
            return coords @ rotmat

    def sort(self, sort_by: Union[str, np.ndarray] = 'symbol',
             reverse: bool = False,
             inplace: bool = True) -> Optional[MultiMolecule]:
        """Sort the atoms in **self** and **self.atoms**, performing in inplace update.

        :parameter sort_by: The property which is to be used for sorting. Accepted values:
            **symbol** (*i.e.* alphabetical), **atnum**, **mass**, **radius** or
            **connectors**. See the plams.PeriodicTable_ module for more details.
            Alternatively, a user-specified array of indices can be provided for sorting.
        :type sort_by: |str|_ or |np.ndarray|_ [|np.int64|_]
        :parameter bool reverse: Sort in reversed order.
        """
        # Create and, potentially, sort a list of indices
        if isinstance(sort_by, str):
            sort_by_array = self._get_atomic_property(prop=sort_by)
            _idx_range = range(self.shape[0])
            idx_range = np.array([i for _, i in sorted(zip(sort_by_array, _idx_range))])
        else:
            assert sort_by.shape[0] == self.shape[1]
            idx_range = sort_by

        # Reverse or not
        if reverse:
            idx_range.reverse()

        # Sort **self**
        self[:] = self[:, idx_range]

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

    def residue_argsort(self, concatenate: bool = True) -> Union[List[List[int]], np.ndarray]:
        """Return the indices that would sort **self** by residue number.

        Residues are defined based on moleculair fragments based on **self.bonds**.

        :parameter bool concatenate: If False, returned a nested list with atomic indices. Each
            sublist contains the indices of a single residue.
        :return: An array of indices that would sort :math:`n` atoms **self**.
        :rtype: :math:`n` |np.ndarray|_ [|np.int64|_].
        """
        # Define residues
        plams_mol = self.as_Molecule(mol_subset=0)[0]
        frags = plams_mol.separate_mod()
        symbol = self.symbol

        # Sort the residues
        core = []
        ligands = []
        for frag in frags:
            if len(frag) == 1:
                core += frag
            else:
                i = np.array(frag)
                argsort = np.argsort(symbol[i])
                ligands.append(i[argsort].tolist())
        core.sort()
        ligands.sort()

        ret = [core] + ligands
        if concatenate:
            return np.concatenate(ret)
        return ret

    def get_center_of_mass(self, mol_subset: MolSubset = None,
                           atom_subset: AtomSubset = None) -> np.ndarray:
        """Get the center of mass.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all :math:`m` molecules in **self** if
            *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :return: An array with the centres of mass of :math:`m` molecules with :math:`n` atoms.
        :rtype: :math:`m*3` |np.ndarray|_ [|np.float64|_].
        """
        coords = self.as_mass_weighted(mol_subset, atom_subset)
        return coords.sum(axis=1) / self.mass.sum()

    def get_bonds_per_atom(self, atom_subset: AtomSubset = None) -> np.ndarray:
        """Get the number of bonds per atom in **self**.

        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :return: An array with the number of bonds per atom, for all :math:`n` atoms in **self**.
        :rtype: :math:`n` |np.ndarray|_ [|np.int64|_].
        """
        j = self._get_atom_subset(atom_subset)
        if self.bonds is None:
            return np.zeros(len(j), dtype=int)
        return np.bincount(self.bonds[:, 0:2].flatten(), minlength=self.shape[1])[j]

    """################################## Root Mean Squared ################################## """

    def _get_time_averaged_prop(self, method: Callable,
                                atom_subset: AtomSubset = None,
                                kwarg: dict = {}) -> pd.DataFrame:
        """A method for constructing time-averaged properties."""
        # Prepare arguments
        at_subset = atom_subset or tuple(self.atoms)
        loop = self._get_loop(atom_subset)

        # Get the time-averaged property
        if loop:
            data = [method(atom_subset=at, **kwarg) for at in at_subset]
        else:
            data = method(atom_subset=at_subset, **kwarg)

        # Construct and return the dataframe
        idx_range = np.arange(0, self.shape[1])
        idx = pd.Index(idx_range, name='Abritrary atomic index')
        column_range, data = self._get_rmsf_columns(data, idx, loop=loop, atom_subset=at_subset)
        columns = pd.Index(column_range, name='Atoms')
        return pd.DataFrame(data, index=idx, columns=columns)

    def _get_average_prop(self, method: Callable,
                          atom_subset: AtomSubset = None,
                          kwarg: Dict[str, Any] = {}) -> pd.DataFrame:
        """A method used for constructing averaged properties."""
        # Prpare arguments
        at_subset = atom_subset or tuple(self.atoms)
        loop = self._get_loop(atom_subset)

        # Calculate and averaged property
        if loop:
            data = np.array([method(atom_subset=at, **kwarg) for at in at_subset]).T
        else:
            data = method(atom_subset=atom_subset, **kwarg).T

        # Construct and return the dataframe
        column_range = self._get_rmsd_columns(loop, atom_subset)
        columns = pd.Index(column_range, name='Atoms')
        return pd.DataFrame(data, columns=columns)

    def init_average_velocity(self, timestep: float = 1.0,
                              rms: bool = False,
                              mol_subset: MolSubset = None,
                              atom_subset: AtomSubset = None) -> pd.DataFrame:
        """Calculate the average velocty (in fs/A) for all atoms in **atom_subset** over the
        course of a trajectory.

        The velocity is averaged over all atoms in a particular atom subset.

        :parameter float timestep: The stepsize, in femtoseconds, between subsequent frames.
        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all :math:`m` molecules in **self** if
            *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :return: A dataframe holding :math:`m-1` velocities averaged over one or more atom subsets.
        :rtype: |pd.DataFrame|_ (values: |np.float64|_)
        """
        kwarg = {'mol_subset': mol_subset, 'timestep': timestep, 'rms': rms}
        df = self._get_average_prop(self.get_average_velocity, atom_subset, kwarg)
        df.index.name = 'Time / fs'
        return df

    def init_time_averaged_velocity(self, timestep: float = 1.0,
                                    rms: bool = False,
                                    mol_subset: MolSubset = None,
                                    atom_subset: AtomSubset = None) -> pd.DataFrame:
        """Calculate the time-averaged velocty (in fs/A) for all atoms in **atom_subset** over the
        course of a trajectory.

        :parameter float timestep: The stepsize, in femtoseconds, between subsequent frames.
        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all :math:`m` molecules in **self** if
            *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol. Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :return: A dataframe holding :math:`m-1` velocities averaged over one or more atom subsets.
        :rtype: |pd.DataFrame|_ (values: |np.float64|_)
        """
        kwarg = {'mol_subset': mol_subset, 'timestep': timestep, 'rms': rms}
        return self._get_time_averaged_prop(self.get_time_averaged_velocity, atom_subset, kwarg)

    def init_rmsd(self, mol_subset: MolSubset = None,
                  atom_subset: AtomSubset = None,
                  reset_origin: bool = True) -> pd.DataFrame:
        """Initialize the RMSD calculation, returning a dataframe.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all :math:`m` molecules in **self** if
            *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset:  |None|_, |int|_ or |str|_
        :parameter bool reset_origin: Reset the origin of each molecule in **self** by means of
            a partial Procrustes superimposition, translating and rotating the molecules.
        :return: A dataframe of RMSDs with one column for every string or list of ints in
            **atom_subset**. Keys consist of atomic symbols (*e.g.* 'Cd') if **atom_subset**
            contains strings, otherwise a more generic 'series ' + str(int) scheme is adopted
            (*e.g.* 'series 2'). Molecular indices are used as indices.
        :rtype: |pd.DataFrame|_ (keys: |str|_, values: |np.float64|_, indices: |np.int64|_).
        """
        if reset_origin:
            self.reset_origin()
        kwarg = {'mol_subset': mol_subset}
        df = self._get_average_prop(self.get_rmsd, atom_subset, kwarg)
        df.index.name = 'XYZ frame number'
        return df

    def init_rmsf(self, mol_subset: MolSubset = None,
                  atom_subset: AtomSubset = None,
                  reset_origin: bool = True) -> pd.DataFrame:
        """Initialize the RMSF calculation, returning a dataframe.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all :math:`m` molecules in **self** if
            *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset:  |None|_, |int|_ or |str|_
        :parameter bool reset_origin: Reset the origin of each molecule in **self** by means of
            a partial Procrustes superimposition, translating and rotating the molecules.
        :return: A dataframe of RMSFs with one column for every string or list of ints in
            **atom_subset**. Keys consist of atomic symbols (*e.g.* 'Cd') if **atom_subset**
            contains strings, otherwise a more generic 'series ' + str(int) scheme is adopted
            (*e.g.* 'series 2'). Molecular indices are used as indices.
        :rtype: |pd.DataFrame|_ (keys: |str|_, values: |np.float64|_, indices: |np.int64|_).
        """
        if reset_origin:
            self.reset_origin()
        kwarg = {'mol_subset': mol_subset}
        return self._get_time_averaged_prop(self.get_rmsf, atom_subset, kwarg)

    def get_average_velocity(self, timestep: float = 1.0,
                             rms: bool = False,
                             mol_subset: MolSubset = None,
                             atom_subset: AtomSubset = None) -> np.ndarray:
        """Return the mean or root-mean squared velocity."""
        if not rms:
            return self.get_velocity(timestep, mol_subset=mol_subset,
                                     atom_subset=atom_subset).mean(axis=1)
        else:
            v = self.get_velocity(timestep, mol_subset=mol_subset, atom_subset=atom_subset)
            return MultiMolecule(v, self.atoms).get_rmsd(mol_subset)

    def get_time_averaged_velocity(self, timestep: float = 1.0,
                                   rms: bool = False,
                                   mol_subset: MolSubset = None,
                                   atom_subset: AtomSubset = None) -> np.ndarray:
        """Return the mean or root-mean squared velocity (mean = time-averaged)."""
        if not rms:
            return self.get_velocity(timestep, mol_subset, atom_subset).mean(axis=0)
        else:
            v = self.get_velocity(timestep, mol_subset, atom_subset)
            return MultiMolecule(v, self.atoms).get_rmsf(mol_subset)

    def get_velocity(self, timestep: float = 1.0,
                     norm: bool = True,
                     mol_subset: MolSubset = None,
                     atom_subset: AtomSubset = None) -> np.ndarray:
        """Calculate the velocty (in fs/A) for all atoms in **atom_subset** over the course of a
        trajectory.

        :parameter float timestep: The stepsize, in femtoseconds, between subsequent frames.
        :parameter float norm: Return the velocity as the norm of the :math:`x`, :math:`y`
            and :math:`z` components.
        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all :math:`m` molecules in **self** if
            *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :return: A 3D array with :math:`m-1` velocities of :math:`n` atoms.
        :rtype: :math:`(m-1)*n` or :math:`(m-1)*n*3` |np.ndarray|_ [|np.float64|_]
        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Slice the XYZ array
        xyz_slice = self[i, j]
        dim1, dim2, dim3 = xyz_slice.shape
        shape = (dim1 - 1) * dim2, dim3

        # Reshape the XYZ array
        A = xyz_slice[:-1].reshape(shape)
        B = xyz_slice[1:].reshape(shape)

        # Calculate and return the velocity
        if norm:
            v = np.linalg.norm(A - B, axis=1)
            v.shape = (dim1 - 1), dim2
        else:
            v = A - B
            v.shape = (dim1 - 1), dim2, 3
        return v

    def get_rmsd(self, mol_subset: MolSubset = None,
                 atom_subset: AtomSubset = None) -> np.ndarray:
        """Calculate the root mean square displacement (RMSD) with respect to the first molecule
        **self** AKA the root mean square of the average nuclear displacement.

        Returns a dataframe with the RMSD as a function of the XYZ frame numbers.
        """
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Calculate and return the RMSD per molecule in **self**
        dist = np.linalg.norm(self[i, j, :] - self[0, j, :], axis=2)
        return np.sqrt(np.einsum('ij,ij->i', dist, dist) / dist.shape[1])

    def get_rmsf(self, mol_subset: MolSubset = None,
                 atom_subset: AtomSubset = None) -> np.ndarray:
        """Calculate the root mean square fluctuation (RMSF) of **self**, AKA the root mean square
        of the time-averaged nuclear displacement.
        Returns a dataframe as a function of atomic indices."""
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Calculate the RMSF per molecule in **self**
        mean_coords = np.mean(self[i, j, :], axis=0)[None, ...]
        displacement = np.linalg.norm(self[i, j, :] - mean_coords, axis=2)**2
        return np.mean(displacement, axis=0)

    @staticmethod
    def _get_rmsd_columns(loop: bool,
                          atom_subset: AtomSubset = None) -> Container[Hashable]:
        """Return the columns for the RMSD dataframe."""
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

    def _get_rmsf_columns(self, rmsf: np.ndarray,
                          index: Container[Hashable],
                          loop: bool,
                          atom_subset: AtomSubset = None
                          ) -> Tuple[Container[Hashable], np.ndarray]:
        """Return the columns and data for the RMSF dataframe."""
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

    @staticmethod
    def _get_loop(subset: AtomSubset) -> bool:
        """Figure out if the supplied subset warrants a for loop or not."""
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
            return False  # subset is an iterable of *int*
        elif isinstance(subset[0][0], int):
            return True  # subset is a nested iterable of *int*
        raise TypeError()

    """#############################  Determining shell structures  ######################### """

    def init_shell_search(self, mol_subset: MolSubset = None,
                          atom_subset: AtomSubset = None,
                          rdf_cutoff: float = 0.5
                          ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Calculate and return properties which can help determining shell structures.

        The following two properties are calculated and returned:

        * The mean distance (per atom) with respect to the center of mass (*i.e.* a modified RMSF).
        * A series mapping abritrary atomic indices in the RMSF to the actual atomic indices.
        * The radial distribution function (RDF) with respect to the center of mass.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index. Include all :math:`m` molecules in **self** if
            *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.  Include all :math:`n` atoms per
            molecule in **self** if *None*.
        :type atom_subset:  |None|_, |int|_ or |str|_
        :parameter float rdf_cutoff: Remove all values in the RDF below this value (Angstrom).
            Usefull for dealing with divergence as the "inter-atomic" distance approaches 0.0 A.
        :return: Returns the following items:

            1. A dataframe holding the mean distance of all atoms with respect the to center
            of mass.

            2. A series mapping the indices from 1. to the actual atomic indices.

            3. A dataframe holding the RDF with respect to the center of mass.

        :rtype: |pd.DataFrame|_, |pd.Series|_ and |pd.DataFrame|_
        """
        def _get_mean_dist(mol_cp, at):
            ret = np.linalg.norm(mol_cp[:, mol_cp.atoms[at]], axis=2).mean(axis=0)
            at_idx = np.argsort(ret)
            return at_idx, sorted(ret)

        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        at_subset = atom_subset or tuple(self.atoms.keys())

        # Calculate the mean distance (per atom) with respect to the center of mass
        # Conceptually similar an RMSF, the "fluctuation" being with respect to the center of mass
        dist_mean = []
        mol_cp = self.copy()[i]
        mol_cp -= mol_cp.get_center_of_mass()[:, None, :]
        at_idx, dist_mean = zip(*[_get_mean_dist(mol_cp, at) for at in at_subset])

        # Create Series containing the actual atomic indices
        at_idx = list(chain.from_iterable(at_idx))
        idx_series = pd.Series(np.arange(0, self.shape[1]), name='Actual atomic index')
        idx_series.loc[0:len(at_idx)-1] = at_idx
        idx_series.index.name = 'Arbitrary atomic index'

        # Cast the modified RMSF results in a dataframe
        index = np.arange(0, self.shape[1])
        kwarg = {'loop': True, 'atom_subset': at_subset}
        columns, data = mol_cp._get_rmsf_columns(dist_mean, index, **kwarg)
        rmsf = pd.DataFrame(data, columns=columns, index=index)
        rmsf.columns.name = 'Distance from origin\n  /  Ångström'
        rmsf.index.name = 'Arbitrary atomic index'

        # Calculate the RDF with respect to the center of mass
        at_dummy = np.zeros_like(mol_cp[:, 0, :])[:, None, :]
        mol_cp = MultiMolecule(np.hstack((mol_cp, at_dummy)), atoms=mol_cp.atoms)
        mol_cp.atoms['origin'] = [mol_cp.shape[1] - 1]
        at_subset = ('origin', ) + at_subset
        with np.errstate(divide='ignore', invalid='ignore'):
            rdf = mol_cp.init_rdf(at_subset)
        del rdf['origin origin']
        rdf = rdf.loc[rdf.index >= rdf_cutoff, [i for i in rdf.columns if 'origin' in i]]

        return rmsf, idx_series, rdf

    @staticmethod
    def get_at_idx(rmsf: pd.DataFrame,
                   idx_series: pd.Series,
                   dist_dict: Dict[str, List[int]]) -> Dict[str, List[int]]:
        """Create subsets of atomic indices (using **rmsf** and **idx_series**) based on
        distance criteria in **dist_dict**.


        For example, ``dist_dict = {'Cd': [3.0, 6.5]}`` will create and return a dictionary with
        three keys: One for all atoms whose RMSF is smaller than 3.0, one where the RMSF is
        between 3.0 and 6.5, and finally one where the RMSF is larger than 6.5.

        This example is illustrated below:

        .. code:: python

            >>> dist_dict = {'Cd': [3.0, 6.5]}
            >>> idx_series = pd.Series(np.arange(12))
            >>> rmsf = pd.DataFrame({'Cd': np.arange(12, dtype=float)})
            >>> get_at_idx(rmsf, idx_series, dist_dict)

            {'Cd_1': [0, 1, 2],
             'Cd_2': [3, 4, 5],
             'Cd_3': [7, 8, 9, 10, 11]
            }

        :parameter rmsf: A dataframe holding the results of an RMSF calculation.
        :type rmsf: |pd.DataFrame|_ (values: |np.int64|_, index: |pd.Int64Index|_)
        :parameter idx_series: A series mapping the indices from **rmsf** to actual atomic indices.
        :type idx_series: |pd.Series|_ (values: |np.int64|_, index: |pd.Int64Index|_)
        :parameter dist_dict: A dictionary with atomic symbols (see **rmsf.columns**) and a
            list of interatomic distances.
        :type min_dict: |dict|_ (keys: |str|_, values: |list|_ [|int|_])
        :return: A dictionary with atomic symbols as keys, and matching atomic indices as values.
        :rtype: |dict|_ (keys: |str|_, values: |list|_ [|int|_])
        """
        # Double check if all keys in **dist_dict** are available in **rmsf.columns**
        for key in dist_dict:
            if key not in rmsf:
                raise KeyError(key, 'was found in "dist_dict" yet is absent from "rmsf"')

        ret = {}
        for key, value in rmsf.items():
            try:
                dist_range = sorted(dist_dict[key])
            except KeyError:
                dist_range = [np.inf]
            dist_min = 0.0
            name = key + '_{:d}'

            for i, dist_max in enumerate(dist_range, 1):
                idx = rmsf[(value >= dist_min) & (value < dist_max)].index
                if idx.any():
                    ret[name.format(i)] = sorted(idx_series[idx].values.tolist())
                dist_min = dist_max

            idx = rmsf[(rmsf[key] > dist_max)].index
            if idx.any():
                ret[name.format(i+1)] = sorted(idx_series[idx].values.tolist())
        return ret

    """#############################  Radial Distribution Functions  ######################### """

    def init_rdf(self, atom_subset: AtomSubset = None,
                 dr: float = 0.05,
                 r_max: float = 12.0,
                 low_mem: bool = False):
        """Initialize the calculation of radial distribution functions (RDFs).

        RDFs are calculated for all possible atom-pairs in **atom_subset** and returned as a
        dataframe.

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
        at_subset = atom_subset or tuple(self.atoms.keys())
        atom_pairs = self.get_pair_dict(at_subset, r=2)

        # Construct an empty dataframe with appropiate dimensions, indices and keys
        df = get_rdf_df(atom_pairs, dr, r_max)
        kwarg = {'dr': dr, 'r_max': r_max}

        # Fill the dataframe with RDF's, averaged over all conformations in **self**
        if low_mem:  # Slower low memory approach
            for i in range(self.shape[0]):
                for key, at in atom_pairs.items():
                    dist_mat = self.get_dist_mat(mol_subset=i, atom_subset=at)
                    df[key] += get_rdf_lowmem(dist_mat, **kwarg)
            df.loc[0.0] = 0.0
            df /= self.shape[0]

        else:  # Faster high memory approach
            for key, at in atom_pairs.items():
                dist_mat = self.get_dist_mat(atom_subset=at)
                df[key] = get_rdf(dist_mat, **kwarg)

        return df

    def get_dist_mat(self, mol_subset: MolSubset = None,
                     atom_subset: Tuple[AtomSubset] = (None, None)) -> np.ndarray:
        """Create and return a distance matrix for all molecules and atoms in **self**.

        Returns a 3D array.

        :parameter mol_subset: Create a distance matrix from a subset of :math:`m` molecules in
            **self**. If *None*, create a distance matrix for all molecules in **self**.
        :type mol_subset: |None|_ or |tuple|_ [|int|_]
        :parameter atom_subset: Create a distance matrix from a subset of atoms per molecule in
            **self**. Values have to be supplied for all 2 dimensions. Atomic indices
            (on or multiple), atomic symbols (one or multiple) and *None* can be freely mixed.
            If *None*, pick all atoms from **self** for that partical dimension; if an
            atomic symbol, do the same for all indices associated with that particular symbol.
        :type atom_subset: |tuple|_ [|None|_], |tuple|_ [|int|_]
        :return: A 3D distance matrix of :math:`m` molecules, created out of two sets of :math:`n`
            and :math:`k` atoms.
        :return type: :math:`m*n*k` |np.ndarray|_ [|np.float64|_].
        """
        # Define array slices
        m_subset = self._get_mol_subset(mol_subset)
        i = m_subset, self._get_atom_subset(atom_subset[0])
        j = m_subset, self._get_atom_subset(atom_subset[1])

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

    @staticmethod
    def get_pair_dict(atom_subset: AtomSubset,
                      r: int = 2) -> Dict[str, str]:
        """Take a subset of atoms and return a dictionary.

        :parameter atom_subset: A subset of atoms.
        :parameter int r: The length of the to-be returned subsets.
        """
        values = list(combinations_with_replacement(atom_subset, r))

        if not isinstance(atom_subset[0], str):
            str_ = 'series' + ''.join(' {:d}' for _ in values[0])
            return {str_.format(*[i.index(j) for j in i]): i for i in values}

        else:
            str_ = ''.join(' {}' for _ in values[0])[1:]
            return {str_.format(*i): i for i in values}

    """############################  Angular Distribution Functions  ######################### """

    def init_adf(self, atom_subset: AtomSubset = None,
                 low_mem: bool = True) -> pd.DataFrame:
        """Initialize the calculation of angular distribution functions (ADFs).

        ADFs are calculated for all possible atom-pairs in **atom_subset** and returned as a
        dataframe.

        :parameter atom_subset: A tuple of atomic symbols. RDFs will be calculated for all
            possible atom-pairs in **atoms**. If *None*, calculate RDFs for all possible atom-pairs
            in the keys of **self.atoms**.
        :type atom_subset: |None|_ or |tuple|_ [|str|_]
        :parameter bool low_mem: If *True*, use a slower but more memory efficient method for
            constructing the ADFs. WARNING: Constructing ADFs is significantly more memory intensive
            than ADFs and in most cases it is recommended to keep this argument at *False*.
        :return: A dataframe of angular distribution functions, averaged over all conformations in
            **self**.
        :rtype: |pd.DataFrame|_ (keys: |str|_, values: |np.float64|_, indices: |np.float64|_).
        """
        # If **atom_subset** is None: extract atomic symbols from they keys of **self.atoms**
        at_subset = atom_subset or tuple(self.atoms.keys())
        atom_pairs = self.get_pair_dict(at_subset, r=3)

        # Construct an empty dataframe with appropiate dimensions, indices and keys
        df = get_adf_df(atom_pairs)

        # Fill the dataframe with RDF's, averaged over all conformations in **self**
        if low_mem:  # Slower low memory approach
            for i in range(self.shape[0]):
                for key, at in atom_pairs.items():
                    a_mat, r_max = self.get_angle_mat(atom_subset=at, mol_subset=i, get_r_max=True)
                    df[key] += get_adf(a_mat, r_max=r_max)
            df /= self.shape[0]

        else:  # Faster high memory approach
            for key, at in atom_pairs.items():
                a_mat, r_max = self.get_angle_mat(atom_subset=at, get_r_max=True)
                df[key] = get_adf(a_mat, r_max=r_max)

        return df

    def get_angle_mat(self, mol_subset: MolSubset = 0,
                      atom_subset: Tuple[AtomSubset] = (None, None, None),
                      get_r_max: bool = False) -> np.ndarray:
        """Create and return an angle matrix for all molecules and atoms in **self**.

        Returns a 4D array.

        :parameter mol_subset: Create a distance matrix from a subset of :math:`m` molecules in
            **self**. If *None*, create a distance matrix for all molecules in
            **self**.
        :type mol_subset: |None|_ or |tuple|_ [|int|_]
        :parameter atom_subset: Create a distance matrix from a subset of atoms per molecule in
            **self**. Values have to be supplied for all 3 dimensions. Atomic indices
            (on or multiple), atomic symbols (one or multiple) and *None* can be freely mixed.
            If *None*, pick all atoms from **self** for that partical dimension; if an
            atomic symbol, do the same for all indices associated with that particular symbol.
        :type atom_subset: |None|_ or |tuple|_ [|str|_]
        :parameter bool get_r_max: Whether or not the maximum distance should be returned or not.
        :return: A 4D angle matrix of :math:`m` molecules, created out of three sets of :math:`n`,
            :math:`k` and :math:`l` atoms.
            If **get_r_max** = *True*, also return the maximum distance.
        :return type: :math:`m*n*k*l` |np.ndarray|_ [|np.float64|_] and (optionally) |float|_
        """
        # Define array slices
        m_subset = self._get_mol_subset(mol_subset)
        i = self._get_atom_subset(atom_subset[0])
        j = self._get_atom_subset(atom_subset[1])
        k = self._get_atom_subset(atom_subset[2])

        # Slice and broadcast the XYZ array
        A = self[m_subset][:, i][..., None, :]
        B = self[m_subset][:, j][:, None, ...]
        C = self[m_subset][:, k][:, None, ...]

        # Temporary ignore RuntimeWarnings related to dividing by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            # Prepare the unit vectors
            kwarg1 = {'atom_subset': [atom_subset[0], atom_subset[1]], 'mol_subset': m_subset}
            kwarg2 = {'atom_subset': [atom_subset[0], atom_subset[2]], 'mol_subset': m_subset}
            dist_mat1 = self.get_dist_mat(**kwarg1)[..., None]
            dist_mat2 = self.get_dist_mat(**kwarg2)[..., None]
            r_max = max(dist_mat1.max(), dist_mat2.max())
            unit_vec1 = (B - A) / dist_mat1
            unit_vec2 = (C - A) / dist_mat2

            # Create and return the angle matrix
            if get_r_max:
                return np.arccos(np.einsum('ijkl,ijml->ijkm', unit_vec1, unit_vec2)), r_max
            return np.arccos(np.einsum('ijkl,ijml->ijkm', unit_vec1, unit_vec2))

    def _get_atom_subset(self, subset: AtomSubset) -> Union[slice, Container[int]]:
        """Grab and return a list of indices from **self.atoms**.

        Return *at* if it is *None*, an *int* or iterable container consisting of *int*.
        """
        if subset is None:
            return slice(0, None)
        elif isinstance(subset, slice):
            return subset

        elif isinstance(subset, (int, np.integer)):
            return [subset]
        elif isinstance(subset[0], (int, np.integer)):
            return subset
        elif isinstance(subset, str):
            return self.atoms[subset]
        elif isinstance(subset[0], str):
            return list(chain.from_iterable(self.atoms[i] for i in subset))
        elif isinstance(subset[0][0], (int, np.integer)):
            return list(chain.from_iterable(subset))
        raise TypeError("'{}' is not a supported object type".format(subset.__class__.__name__))

    @staticmethod
    def _get_mol_subset(subset: MolSubset) -> slice:
        """"""
        if subset is None:
            return slice(0, None)
        elif isinstance(subset, slice):
            return subset
        elif isinstance(subset, (int, np.integer)):
            if subset >= 0:
                return slice(subset, subset+1)
            else:
                return slice(subset-1, subset)
        elif len(subset) == 1 and isinstance(subset[0], (int, np.integer)):
            i = subset[0]
            if i >= 0:
                return slice(i, i+1)
            else:
                return slice(i-1, i)
        raise TypeError("'{}' is not a supported object type".format(subset.__class__.__name__))

    """#################################  Type conversion  ################################### """

    def _mol_to_file(self, filename: str,
                     outputformat: Optional[str] = None,
                     mol_subset: MolSubset = 0) -> None:
        """Create files using the plams.Molecule.write_ method.

        :parameter str filename: The path+filename (including extension) of the to be created file.
        :parameter str outputformat: The outputformat; accepated values are *mol*, *mol2*, *pdb* or
            *xyz*.
        :parameter mol_subset: Perform the operation on a subset of molecules in **self**, as
            determined by their moleculair index.
            Include all :math:`m` molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]

        .. _plams.Molecule.write: https://www.scm.com/doc/plams/components/mol_api.html\
    #scm.plams.mol.molecule.Molecule.write
        """
        _m_subset = self._get_mol_subset(mol_subset)
        m_subset = range(_m_subset.start, _m_subset.stop, _m_subset.step)
        outputformat = outputformat or filename.rsplit('.', 1)[-1]
        plams_mol = self.as_Molecule(mol_subset=0)[0]

        if len(m_subset) != 1 or (m_subset.stop - m_subset.start) // m_subset.step != 1:
            name_list = filename.rsplit('.', 1)
            name_list.insert(-1, '.{:d}.')
            name = ''.join(name_list)
        else:
            name = filename

        for i, j in enumerate(m_subset, 1):
            plams_mol.from_array(self[j])
            plams_mol.write(name.format(i), outputformat=outputformat)

    def as_pdb(self, mol_subset: MolSubset = 0,
               filename: str = 'mol.pdb') -> None:
        """Convert a *MultiMolecule* object into one or more Protein DataBank files (.pdb).

        Utilizes the plams.Molecule.write_ method.

        :parameter str filename: The path+filename (including extension) of the to be created file.
        :parameter mol_subset: Perform the operation on a subset of molecules in **self**, as
            determined by their moleculair index.
            Includes all :math:`m` molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]

        .. _plams.Molecule.write: https://www.scm.com/doc/plams/components/mol_api.html\
    #scm.plams.mol.molecule.Molecule.write
        """
        self._mol_to_file(filename, 'pdb', mol_subset)

    def as_mol2(self, mol_subset: MolSubset = 0,
                filename: str = 'mol.mol2') -> None:
        """Convert a *MultiMolecule* object into one or more .mol2 files.

        Utilizes the plams.Molecule.write_ method.

        :parameter str filename: The path+filename (including extension) of the to be created file.
        :parameter mol_subset: Perform the operation on a subset of molecules in **self**, as
            determined by their moleculair index.
            Includes all :math:`m` molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]

        .. _plams.Molecule.write: https://www.scm.com/doc/plams/components/mol_api.html\
    #scm.plams.mol.molecule.Molecule.write
        """
        self._mol_to_file(filename, 'mol2', mol_subset)

    def as_mol(self, mol_subset: MolSubset = 0,
               filename: str = 'mol.mol') -> None:
        """Convert a *MultiMolecule* object into one or more .mol files.

        Utilizes the plams.Molecule.write_ method.

        :parameter str filename: The path+filename (including extension) of the to be created file.
        :parameter mol_subset: Perform the operation on a subset of molecules in **self**, as
            determined by their moleculair index.
            Includes all :math:`m` molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]

        .. _plams.Molecule.write: https://www.scm.com/doc/plams/components/mol_api.html\
    #scm.plams.mol.molecule.Molecule.write
        """
        self._mol_to_file(filename, 'mol', mol_subset)

    def as_xyz(self, mol_subset: MolSubset = None,
               filename: str = 'mol.xyz') -> None:
        """Create an .xyz file out of **self**.

        :parameter str filename: The path+filename (including extension) of the to be created file.
        :parameter mol_subset: Perform the operation on a subset of molecules in **self**, as
            determined by their moleculair index.
            Includes all :math:`m` molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        """
        # Define constants
        m_subset = self._get_mol_subset(mol_subset)
        at = self.symbol[:, None]
        header = '{:d}\nframe '.format(len(at))
        kwarg = {'fmt': ['%-10.10s', '%-15s', '%-15s', '%-15s'],
                 'delimiter': '     ',
                 'comments': ''}

        # Create the .xyz file
        with open(filename, 'wb') as file:
            for i, xyz in enumerate(self[m_subset], 1):
                np.savetxt(file, np.hstack((at, xyz)), header=header+str(i), **kwarg)

    def as_mass_weighted(self, mol_subset: MolSubset = None,
                         atom_subset: AtomSubset = None,
                         inplace: bool = False) -> Optional[MultiMolecule]:
        """Transform the Cartesian of **self** into mass-weighted Cartesian coordinates.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index.
            Include all :math:`m` molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        :parameter bool inplace: Instead of returning the new coordinates, perform an inplace
            update of **self**.
        :return: if **inplace** = *False*: a new :class:`.MultiMolecule` instance with the
            mass-weighted Cartesian coordinates of :math:`m` molecules with :math:`n` atoms.
        :rtype: :math:`m*n*3` |np.ndarray|_ [|np.float64|_]
        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Create an array of mass-weighted Cartesian coordinates
        if inplace:
            self[i, j, :] *= self.mass[None, j, None]
        else:
            return self[i, j, :] * self.mass[None, j, None]

    def from_mass_weighted(self, mol_subset: MolSubset = None,
                           atom_subset: AtomSubset = None) -> None:
        """Transform **self** from mass-weighted Cartesian into Cartesian coordinates.

        Performs an inplace update of **self**.

        :parameter mol_subset: Perform the calculation on a subset of molecules in **self**, as
            determined by their moleculair index.
            Include all :math:`m` molecules in **self** if *None*.
        :type mol_subset: |None|_, |int|_ or |list|_ [|int|_]
        :parameter atom_subset: Perform the calculation on a subset of atoms in **self**, as
            determined by their atomic index or atomic symbol.
            Include all :math:`n` atoms per molecule in **self** if *None*.
        :type atom_subset: |None|_, |int|_ or |str|_
        """
        # Prepare slices
        i = self._get_mol_subset(mol_subset)
        j = self._get_atom_subset(atom_subset)

        # Update **self**
        self[i, j, :] /= self.mass[None, j, None]

    def as_Molecule(self, mol_subset: MolSubset = None,
                    atom_subset: AtomSubset = None) -> List[Molecule]:
        """Convert **self** into a *list* of *plams.Molecule*.

        :parameter mol_subset: Convert a subset of molecules in **self** as based on their
            indices. If *None*, convert all molecules.
        :type mol_subset: |None|_, |int|_ or |tuple|_ [|int|_]
        :parameter atom_subset: Convert a only subset of atoms within each molecule in
            **self**, as based on their indices. If *None*, convert all atoms per molecule.
        :type atom_subset: |None|_, |int|_ or |tuple|_ [|int|_]
        :return: A list of :math:`m` PLAMS molecules.
        :rtype: :math:`m` |list|_ [|plams.Molecule|_].
        """
        m_subset = self._get_mol_subset(mol_subset)
        if atom_subset is None:
            at_subset = np.arange(self.shape[1])
        else:
            at_subset = sorted(self._get_atom_subset(atom_subset))
        at_symbols = self.symbol

        # Construct a template molecule and fill it with atoms
        assert self.atoms is not None
        mol_template = Molecule()
        mol_template.properties = self.properties.copy()
        for i in at_subset:
            atom = Atom(symbol=at_symbols[i])
            mol_template.add_atom(atom)

        # Fill the template molecule with bonds
        if self.bonds.any():
            bond_idx = np.ones(len(self))
            bond_idx[at_subset] += np.arange(len(at_subset))
            for i, j, order in self.bonds:
                if i in at_subset and j in at_subset:
                    at1 = mol_template[int(bond_idx[i])]
                    at2 = mol_template[int(bond_idx[j])]
                    mol_template.add_bond(Bond(atom1=at1, atom2=at2, order=order/10.0))

        # Create copies of the template molecule; update their cartesian coordinates
        ret = []
        for i, xyz in enumerate(self[m_subset]):
            mol = mol_template.copy()
            mol.from_array(xyz[at_subset])
            mol.properties.frame = i
            ret.append(mol)

        return ret

    @classmethod
    def from_Molecule(cls, mol_list: Union[Molecule, Iterable[Molecule]],
                      subset: Container[str] = 'atoms') -> MultiMolecule:
        """Construct a :class:`.MultiMolecule` instance from a PLAMS molecule or
        a list of PLAMS molecules.

        :parameter mol_list: A PLAMS molecule or list of PLAMS molecules.
        :type mol_list: |plams.Molecule|_ or |list|_ [|plams.Molecule|_]
        :parameter subset: Transfer a subset of *plams.Molecule* attributes to **self**. If *None*,
            transfer all attributes. Accepts one or more of the following values as strings:
            *properties*, *atoms* and/or *bonds*.
        :return: A :class:`.MultiMolecule` instance constructed from **mol_list**.
        :rtype: |FOX.MultiMolecule|_
        """
        if isinstance(mol_list, Molecule):
            plams_mol = mol_list
            mol_list = [mol_list]
        else:
            plams_mol = mol_list[0]
        subset = subset or ('atoms', 'bonds', 'properties')

        # Convert coordinates
        coords = np.array([mol.as_array() for mol in mol_list])
        kwarg: dict = {}

        # Convert atoms
        if 'atoms' in subset:
            kwarg['atoms'] = {}
            for i, at in enumerate(plams_mol.atoms):
                try:
                    kwarg['atoms'][at.symbol].append(i)
                except KeyError:
                    kwarg['atoms'][at.symbol] = [i]

        # Convert properties
        if 'properties' in subset:
            kwarg['properties'] = plams_mol.properties.copy()

        # Convert bonds
        if 'bonds' in subset:
            plams_mol.set_atoms_id(start=0)
            kwarg['bonds'] = np.array([(bond.atom1.id, bond.atom2.id, bond.order * 10) for
                                       bond in plams_mol.bonds], dtype=int)
            plams_mol.unset_atoms_id()

        return cls(coords, **kwarg)

    @classmethod
    def from_xyz(cls, filename: str,
                 bonds: Optional[np.ndarray] = None,
                 properties: Optional[dict] = None) -> MultiMolecule:
        """Construct a :class:`.MultiMolecule` instance from a (multi) .xyz file.

        :parameter str filename: The path + filename of an .xyz file
        :parameter bonds: A 2D array with indices of the atoms defining all :math:`k` bonds
            (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
            Stored in the **bonds** attribute.
        :type bonds: |None|_ or :math:`k*3` |np.ndarray|_ [|np.int64|_]
        :parameter properties: A Settings object (subclass of dictionary) intended for storing
            miscellaneous user-defined (meta-)data. Is devoid of keys by default. Stored in the
            **properties** attribute.
        :type properties: |plams.Settings|_
        :return: A :class:`.MultiMolecule` instance constructed from **filename**.
        :rtype: |FOX.MultiMolecule|_
        """
        return cls(*read_multi_xyz(filename), bonds=None, properties=None)

    @classmethod
    def from_kf(cls, filename: str,
                bonds: Optional[np.ndarray] = None,
                properties: Optional[dict] = None) -> MultiMolecule:
        """Construct a :class:`.MultiMolecule` instance from a KF binary file.

        :parameter str filename: The path + filename of an .xyz file
        :parameter bonds: A 2D array with indices of the atoms defining all :math:`k` bonds
            (columns 1 & 2) and their respective bond orders multiplied by 10 (column 3).
            Stored in the **bonds** attribute.
        :type bonds: |None|_ or :math:`k*3` |np.ndarray|_ [|np.int64|_]
        :parameter properties: A Settings object (subclass of dictionary) intended for storing
            miscellaneous user-defined (meta-)data. Is devoid of keys by default. Stored in the
            **properties** attribute.
        :type properties: |plams.Settings|_
        :return: A :class:`.MultiMolecule` instance constructed from **filename**.
        :rtype: |FOX.MultiMolecule|_
        """
        return cls(*read_kf(filename), bonds, properties)
