""" A module with miscellaneous functions. """

__all__ = ['separate_mod']

from scm.plams.core.errors import MoleculeError


def separate_mod(plams_mol):
    """ A modified version of the PLAMS separate()_ function. Separates the molecule into connected
    component as based on its bonds.
    Returns aforementioned components as a nested list of atomic indices.

    :parameter plams_mol: A PLAMS molecule with
    :type plams_mol: |plams.Molecule|_
    :return: A nested list of atomic indices, each sublist representing a set of unconnected
        moleculair fragments
    :rtype: |list|_ [|list|_ [|int|_]]
    """
    if len(plams_mol.bonds) == 0:
        raise MoleculeError('separate_mod: No bonds were found in plams_mol')

    # Mark atoms
    for i, at in enumerate(plams_mol.atoms):
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
    for at in plams_mol.atoms:
        if not at._visited:
            m = []
            dfs(at, m)
            indices.append(m)

    return indices
