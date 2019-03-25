""" A module which expands on the Molecule class of PLAMS. """

__all__ = ['Molecule']

import numpy as np

from scm.plams import Molecule
from scm.plams.core.errors import MoleculeError


class Molecule(Molecule):
    """ A modified version of the PLAMS Molecule_ class, suplemented with a number of
    additional methods. """

    def separate_mod(self):
        """ A modified version of the PLAMS separate()_ function. Separates the molecule into
        connected component as based on its bonds.
        Returns aforementioned components as a nested list of atomic indices.

        :return: A nested list of atomic indices, each sublist representing a set of unconnected
            moleculair fragments.
        :rtype: |list|_ [|list|_ [|int|_]].
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

    def infer_atom_types(self):
        """ Attempt to infer CHARMM atom types based on bonds and
        (formal) atomic charges in **self**. """
        def infer_H(at):
            at_other = at.bonds[0].other_end(at)
            bonds = [bond.order for bond in at_other.bonds]
            if at_other.properties.charge != 0:
                return 'HC'  # H attached to a charged group
            elif at_other.symbol == 'O':
                return 'HO'  # Hydroxide
            elif 1.5 in bonds or 2.0 in bonds:
                return 'HA'  # Aliphatic or aromatic
            else:
                return 'H'

        def infer_C(at):
            bonds = at.bonds
            if len(bonds) == 2:
                return 'CUY1'  # Triple bond
            elif len(bonds) == 3:
                for bond in bonds:
                    if bond.other_end(at).symbol == 'O' and bond.order in (1.5, 2.0):
                        return 'C'  # Carbonyl
                    return 'CUA1'  # Alkene
            elif len(bonds) == 4:
                return 'CT'  # Tetrahedral carbon

        def infer_O(at):
            # Check for charged oxygens
            if at.properties.charge != 0.0:
                return 'OC'

            # Check for water, ethers & alcohols
            elif len(at.bonds) == 2:
                at1, at2 = [bond.other_end(at) for bond in at.bonds]
                if at1.symbol == at2.symbol == 'C':
                    return 'OE'  # Ethers & acetals
                elif 1 in (at1.atnum, at2.atnum):
                    return 'OT'  # Generic hydroxide

            # Check for carbonyls
            elif len(at.bonds) == 1:
                C = at.bonds[0].other_end(at)
                at_list = [bond.other_end(C) for bond in C.bonds]
                at_symbol_other = [at_other.symbol for at_other in at_list]
                if 'N' in at_symbol_other:
                    return 'O'  # Amide
                elif 'H' in at_symbol_other:
                    return 'OA'  # Aldehyde
                elif at_symbol_other[0] == at_symbol_other[1] == 'C':
                    return 'OK'  # Ketone
                elif 'O' in at_symbol_other:
                    for at2 in at_list:
                        if at2.symbol == 'O' and at2 != at:
                            at_symbol_other2 = [bond.other_end(at2).symbol for bond in at2.bonds]
                            if 'H' in at_symbol_other2:
                                return 'OAC'  # Carboxylic acid
                            elif at_symbol_other2[0] == at_symbol_other2[1] == 'C':
                                return 'OS'  # Ester

        ret = np.zeros(len(self.atoms), dtype='<U4')
        for i, at in enumerate(self):
            if at.symbol == 'H':
                ret[i] = infer_H(at)
            elif at.symbol == 'C':
                ret[i] = infer_C(at)
            elif at.symbol == 'O':
                ret[i] = infer_O(at)

        return ret

    def set_atoms_id(self, start=1):
        """ A modified version of the PLAMS set_atoms_id()_ function.
        Equip each atom in **self** of the molecule with the *id* attribute based on enumeration of
        **self.atoms**.

        :parameter bool start: The starting value for the enumeration procedure.
        """
        for i, at in enumerate(self, start):
            at.id = i

    def get_angles(self):
        """ Return an array with the atomic indices defining all angles in **self**.

        :return: An array with atomic indices defining *n* angles.
        :rtype: *n*3* |np.ndarray|_ [|np.int64|_].
        """
        angle = []
        for at2 in self.atoms:
            if len(at2.bonds) >= 2:
                at_other = [bond.other_end(at2) for bond in at2.bonds]
                for i, at1 in enumerate(at_other, 1):
                    for at3 in at_other[i:]:
                        angle.append((at1.id, at2.id, at3.id))

        return np.array(angle, dtype=int)

    def get_dihedrals(self):
        """ Return an array with the atomic indices defining all proper dihedrals in **self**.

        :return: An array with atomic indices defining *n* proper dihedrals.
        :rtype: *n*4* |np.ndarray|_ [|np.int64|_].
        """
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

        return np.array(dihed, dtype=int)

    def get_impropers(self):
        """ Return an array with the atomic indices defining all improper dihedrals in **self**.

        :return: An array with atomic indices defining *n* improper dihedrals.
        :rtype: *n*4* |np.ndarray|_ [|np.int64|_].
        """
        impropers = []
        for at1 in self.atoms:
            order = [bond.order for bond in at1.bonds]
            if len(order) == 3:
                if 2.0 in order or 1.5 in order:
                    at2, at3, at4 = [bond.other_end(at1) for bond in at1.bonds]
                    impropers.append((at1.id, at2.id, at3.id, at4.id))

        return np.array(impropers, dtype=int)
