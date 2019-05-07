""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

import os
from os.path import join

from scm.plams import add_to_class
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

import FOX
from FOX import (ARMC, MultiMolecule, get_template, get_example_xyz)


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'cp2k.ssmp -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


path = '/Users/bvanbeek/Downloads'

# Read the .xyz file and generate
mol = MultiMolecule.from_xyz(get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol.update_atom_type(join(FOX.__path__[0], 'data/formate.str'))
mol.update_atom_charge('Cd', 0.9768)
mol.update_atom_charge('Se', -0.9768)
mol.update_atom_charge('O_1', -0.4704)
mol.update_atom_charge('C_1', 0.4524)
psf = mol.as_psf(join(path, 'mol.psf'), return_blocks=True)

# Prepare the ARMC settings
s = get_template('armc.yaml', join(FOX.__path__[0], 'examples'))
s.job.psf = psf
s.job.molecule = mol
armc = ARMC.from_dict(s)
armc.job.keep_files = True
armc.job.settings.input.motion.md.steps //= 100
armc.job.settings.input.motion.md.time_start_val //= 100
print(armc)

# Start ARMC
try:
    os.remove(join(path, 'ARMC.hdf5'))
except FileNotFoundError:
    pass
# armc.init_armc()
