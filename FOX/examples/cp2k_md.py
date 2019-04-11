""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

from os.path import join

import numpy as np

from scm.plams import add_to_class
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from FOX.classes.monte_carlo import ARMC
from FOX.classes.multi_mol import MultiMolecule
from FOX.examples.example_xyz import get_example_xyz
from FOX.functions.utils import (get_template, dict_to_pandas)
from FOX.functions.cp2k_utils import (set_subsys_kind, set_lennard_jones, set_atomic_charges)


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'mpirun cp2k.popt -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


# Read the .xyz file
path = '/Users/bvanbeek/Downloads'
mol = MultiMolecule(filename=get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol._set_psf_block()
mol._update_atom_type(join(path, 'formate.str'))
df = mol.properties.atoms
df.loc[df['atom name'] == 'Cd', 'charge'] = 2.0
df.loc[df['atom name'] == 'Se', 'charge'] = -2.0
mol.as_psf(join(path, 'qd.psf'))

# Generate (CP2K) job settings
lj_dict = get_template('LJ_potential.yaml')
charge_dict = get_template('atomic_charges.yaml')
s = get_template('md_cp2k.yaml')
s.input.force_eval.subsys.topology = join(path, 'qd.psf')
s.input.force_eval.mm.forcefield.parm_file_name = join(path, 'par_all36_cgenff.prm')
set_subsys_kind(s, df)
lj_dict2 = set_lennard_jones(s, lj_dict)
charge_dict2 = set_atomic_charges(s, charge_dict)

# Generate a series of parameters
param = dict_to_pandas(get_template('param.yaml'), 'param')
param['keys'] = ''
for i, j in param.index:
    if i == 'charge':
        param.at[(i, j), 'keys'] = charge_dict2[j]
    elif i in ('epsilon', 'sigma'):
        param.at[(i, j), 'keys'] = lj_dict2[j]

# Prepare the ARMC settings
carlos = ARMC(np.zeros(10), mol)
carlos.pes.rdf.func = MultiMolecule.init_rdf
carlos.pes.rdf.kwarg = {'atom_subset': ('Cd', 'Se', 'O')}
carlos.pes.rdf.ref = mol.init_rdf(**carlos.pes.rdf.kwarg)
carlos.job.settings = s
carlos.armc.iter_len = 100
carlos.armc.sub_iter_len = 10

# Run ARMC
charge = df['charge'].copy()
charge.index = df['atom type']
charge_tot = charge.sum().round(8)

at_type = 'Cd'
move = 1.9
if at_type == 'Cd':
    charge[charge.index == 'Cd'] = move
    charge[charge.index == 'Se'] = -move
    charge_tot_new = charge.sum().round(8)
    count = len(charge[(charge.index != 'Cd') & (charge.index != 'Se')])
    i = charge_tot_new / count
    charge[(charge.index != 'Cd') & (charge.index != 'Se')] -= i
elif at_type == 'Se':
    charge[charge.index == 'Se'] = move
    charge[charge.index == 'Cd'] = -move
    charge_tot_new = charge.sum().round(8)
    count = len(charge[(charge.index != 'Cd') & (charge.index != 'Se')])
    i = charge_tot_new / count
    charge[(charge.index != 'Cd') & (charge.index != 'Se')] -= i
else:
    charge[charge.index == at_type] = move
    charge_tot_new = charge.sum().round(8)
    count = len(charge[charge.index != at_type])
    i = charge_tot_new / count
    charge[charge.index != at_type] -= i

#carlos.init_armc()
