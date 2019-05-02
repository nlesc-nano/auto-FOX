""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

import os
from os.path import join

from scm.plams import add_to_class
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

import FOX
from FOX import (ARMC, MultiMolecule, get_template, get_example_xyz)
from FOX.functions.read_prm import (read_prm, write_prm, rename_atom_types)


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'cp2k.ssmp -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


path = '/Users/basvanbeek/Downloads'

# Rename atom types
rename_dict = {'CG2O3': 'C_1', 'HGR52': 'H_1', 'OG2D2': 'O_1'}
prm_dict = read_prm(join(path, 'par_all36_cgenff.prm'))
rename_atom_types(prm_dict, rename_dict)
write_prm(prm_dict, join(path, 'charmm.prm'))

# Read the .xyz file and generate
mol = MultiMolecule.from_xyz(get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol.update_atom_type(join(FOX.__path__[0], 'data/formate.str'))
mol.update_atom_charge('Cd', 2.0)
mol.update_atom_charge('Se', -2.0)
psf = mol.as_psf(join(path, 'mol.psf'), return_blocks=True)
for key, value in rename_dict.items():
    psf['atoms'].loc[psf['atoms']['atom type'] == key, 'atom type'] = value

# Prepare the ARMC settings
s = get_template('armc.yaml')
s.job.psf = psf
s.job.molecule = mol
s.job.path = path
s.hdf5_file = join(path, s.hdf5_file)
carlos = ARMC.from_dict(s)
carlos.job.settings.input.force_eval.mm.forcefield.parm_file_name = join(path, 'charmm.prm')
carlos.job.settings.input.force_eval.subsys.topology.conn_file_name = join(path, 'mol.psf')
carlos.job.keep_files = True

# Start ARMC
try:
    os.remove(join(path, 'ARMC.hdf5'))
except FileNotFoundError:
    pass
# carlos.init_armc()
