""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

import os
from os.path import join

import numpy as np

from scm.plams import Settings, add_to_class
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from FOX.classes.monte_carlo import ARMC
from FOX.classes.multi_mol import MultiMolecule
from FOX.functions.utils import (get_template, dict_to_pandas, get_example_xyz)
from FOX.functions.cp2k_utils import (set_subsys_kind, set_keys)


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'cp2k.ssmp -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


path = '/Users/bvanbeek/Downloads'

# Read the .xyz file
mol = MultiMolecule.from_xyz(get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol.generate_psf_block()
mol.update_atom_type(join(path, 'formate.str'))
df = mol.properties.psf
df.loc[df['atom name'] == 'Cd', 'charge'] = 2.0
df.loc[df['atom name'] == 'Se', 'charge'] = -2.0

# Generate (CP2K) job settings
s = get_template('md_cp2k.yaml')
s.input.force_eval.subsys.topology.conn_file_name = join(path, 'qd.psf')
s.input.force_eval.mm.forcefield.parm_file_name = join(path, 'par_all36_cgenff.prm')
set_subsys_kind(s, df)

# Generate a dataframe of parameters
param = dict_to_pandas(get_template('param.yaml'), 'param')
param['key'] = set_keys(s, param)
param['param_old'] = np.nan

# Set charge constraints
charge_constrain = Settings()
charge_constrain.Cd = {'Se': -1, 'OG2D2': -0.5}
charge_constrain.Se = {'Cd': -1, 'OG2D2': 0.5}
charge_constrain.OG2D2 = {'Cd': -2, 'Se': 2}

# Prepare the ARMC settings
carlos = ARMC(param, mol)
carlos.hdf5_path = path
carlos.job.path = path
carlos.job.settings = s
carlos.job.settings.input.motion.md.steps //= 100
carlos.job.settings.input.motion.md.time_start_val //= 100
carlos.job.settings.input.force_eval.mm.forcefield.spline.emax_spline = 1.0
carlos.job.keep_files = True
carlos.job.psf = mol.as_psf(join(path, 'qd.psf'), return_blocks=True)

carlos.pes.rdf.func = MultiMolecule.init_rdf
carlos.pes.rdf.kwarg = {'atom_subset': ('Cd', 'Se', 'O')}
carlos.pes.rdf.ref = mol.init_rdf(**carlos.pes.rdf.kwarg)

carlos.armc.iter_len = 100
carlos.armc.sub_iter_len = 10

try:
    os.remove(join(path, 'MC.hdf5'))
except FileNotFoundError:
    pass
carlos.init_armc()
