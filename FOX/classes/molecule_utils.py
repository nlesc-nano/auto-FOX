""" A module which expands on the Molecule class of PLAMS. """

__all__ = ['Molecule']

import numpy as np

from scm.plams import Molecule
from scm.plams.core.errors import MoleculeError


class Molecule(Molecule):
    """ A modified version of the PLAMS Molecule_ class, suplemented with a number of
    additional methods. """

    def separate_mod(self):
        """ A modified version of the PLAMS Molecule.separate_ method. Separates the molecule into
        connected component as based on its bonds.
        Returns aforementioned components as a nested list of atomic indices.

        :return: A nested list of atomic indices, each sublist representing a set of unconnected
            moleculair fragments.
        :rtype: |list|_ [|list|_ [|int|_]].
        _Molecule.separate: https://www.scm.com/doc/plams/components/molecule.html#scm.plams.core.basemol.Molecule.separate
        """
        if len(self.bonds) == 0:
            raise MoleculeError('separate_mod: No bonds were found in plams_mol')

        # Mark atoms
        for i, at in enumerate(self.atoms):
            at.id = i
            at._visited = False

        # Loop through atoms
        def dfs(at1, m):
            at1._visited = True
            m.append(at1.id)
            for bond in at1.bonds:
                at2 = bond.other_end(at1)
                if not at2._visited:
                    dfs(at2, m)

        # Create a nested list of atomic indices
        indices = []
        for at in self.atoms:
            if not at._visited:
                m = []
                dfs(at, m)
                indices.append(m)

        return indices

    def fix_bond_orders(self):
        """ Attempt to fix bond orders and (formal) atomic charges in **self**. """
        # Set default atomic charges
        for at in self.atoms:
            if not at.properties.charge:
                at.properties.charge = 0

        # Fix atomic charges and bond orders
        for b1 in self.bonds:
            at1, at2 = b1
            at1_saturation = sum([b2.order for b2 in at1.bonds])
            at1_saturation += -1 * at1.properties.charge - at1.connectors
            at2_saturation = sum([b3.order for b3 in at2.bonds])
            at2_saturation += -1 * at2.properties.charge - at2.connectors
            if at1_saturation == at2_saturation != 0:
                b1.order += np.abs(at1_saturation)
            else:
                if at1_saturation != 0:
                    at1.properties.charge += at1_saturation
                if at2_saturation != 0:
                    at2.properties.charge += at2_saturation

    def set_atoms_id(self, start=1):
        """ A modified version of the PLAMS Molecule.set_atoms_id_ method.
        Equip each atom in **self** of the molecule with the *id* attribute based on enumeration of
        **self.atoms**.

        :parameter bool start: The starting value for the enumeration procedure.
        _Molecule.set_atoms_id: https://www.scm.com/doc/plams/components/molecule.html#scm.plams.core.basemol.Molecule.set_atoms_id
        """
        for i, at in enumerate(self, start):
            at.id = i

    def get_angles(self):
        """ Return an array with the atomic indices defining all angles in **self**.

        :return: An array with atomic indices defining *n* angles.
        :rtype: *n*3* |np.ndarray|_ [|np.int64|_].
        """
        self.set_atoms_id(start=0)
        angle = []
        for at2 in self.atoms:
            if len(at2.bonds) >= 2:
                at_other = [bond.other_end(at2) for bond in at2.bonds]
                for i, at1 in enumerate(at_other, 1):
                    for at3 in at_other[i:]:
                        angle.append((at1.id, at2.id, at3.id))

        return np.array(angle, dtype=int) + 1

    def get_dihedrals(self):
        """ Return an array with the atomic indices defining all proper dihedrals in **self**.

        :return: An array with atomic indices defining *n* proper dihedrals.
        :rtype: *n*4* |np.ndarray|_ [|np.int64|_].
        """
        self.set_atoms_id(start=0)
        dihed = []
        for b1 in self.bonds:
            if len(b1.atom1.bonds) > 1 and len(b1.atom2.bonds) > 1:
                at2, at3 = b1
                for b2 in at2.bonds:
                    at1 = b2.other_end(at2)
                    if at1 != at3:
                        for b3 in at3.bonds:
                            at4 = b3.other_end(at3)
                            if at4 != at2:
                                dihed.append((at1.id, at2.id, at3.id, at4.id))

        return np.array(dihed, dtype=int) + 1

    def get_impropers(self):
        """ Return an array with the atomic indices defining all improper dihedrals in **self**.

        :return: An array with atomic indices defining *n* improper dihedrals.
        :rtype: *n*4* |np.ndarray|_ [|np.int64|_].
        """
        self.set_atoms_id(start=0)
        impropers = []
        for at1 in self.atoms:
            order = [bond.order for bond in at1.bonds]
            if len(order) == 3:
                if 2.0 in order or 1.5 in order:
                    at2, at3, at4 = [bond.other_end(at1) for bond in at1.bonds]
                    impropers.append((at1.id, at2.id, at3.id, at4.id))

        return np.array(impropers, dtype=int) + 1
