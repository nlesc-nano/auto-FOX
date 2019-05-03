""" A Recipe for MM-MD parameter optimizations with CP2K 3.1 or 6.1. """

from os.path import join

import FOX
from FOX import (ARMC, MultiMolecule, get_template, get_example_xyz)


# Read the .xyz file and generate
str_file = join(FOX.__path__[0], 'data/formate.str')
mol = MultiMolecule.from_xyz(get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol.update_atom_type(str_file)
mol.update_atom_charge('Cd', 2.0)
mol.update_atom_charge('Se', -2.0)
psf = mol.as_psf(return_blocks=True)

# Prepare the ARMC settings
path = join(FOX.__path__[0], 'examples')
s = get_template('armc.yaml', path)
s.job.psf = psf
s.job.molecule = mol

# Start the MC parameterization
armc = ARMC.from_dict(s)
# armc.init_armc()
