""" A Recipe for MM-MD parameter optimizations with CP2K 3.1 or 6.1. """

from FOX import (ARMC, MultiMolecule)
from FOX.functions.utils import (get_template, get_example_xyz)


# Read the .xyz file and generate
mol = MultiMolecule.from_xyz(get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol.update_atom_type('formate.str')
mol.update_atom_charge('Cd', 2.0)
mol.update_atom_charge('Se', -2.0)
psf = mol.as_psf(return_blocks=True)

# Prepare the ARMC settings
s = get_template('armc.yaml')
s.job.psf = psf
s.job.molecule = mol

# Start the MC parameterization
armc = ARMC.from_dict(s)
armc.init_armc()
