""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

from os.path import join

from scm.plams import add_to_class
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from FOX.classes.monte_carlo import ARMC
from FOX.classes.multi_mol import MultiMolecule
from FOX.functions.utils import (get_example_xyz, get_template)


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'cp2k.ssmp -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


path = '/Users/basvanbeek/Downloads'

# Read the .xyz file and generate
mol = MultiMolecule.from_xyz(get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol.update_atom_type(join(path, 'formate.str'))
mol.update_atom_charge('Cd', 2.0)
mol.update_atom_charge('Se', -2.0)
psf = mol.as_psf(return_blocks=True)

# Prepare the ARMC settings
s = get_template('armc.yaml')
s.job.psf = psf
s.job.molecule = mol
carlos = ARMC.from_dict(s)
