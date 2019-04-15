""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

from os.path import join

import pandas as pd

from scm.plams import Settings

from FOX.classes.monte_carlo import ARMC
from FOX.classes.multi_mol import MultiMolecule
from FOX.examples.example_xyz import get_example_xyz
from FOX.functions.utils import (get_template, dict_to_pandas, template_to_df)
from FOX.functions.cp2k_utils import (set_subsys_kind, set_lennard_jones, set_atomic_charges)


# Read the .xyz file
path = '/Users/basvanbeek/Downloads'
mol = MultiMolecule(filename=get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol._set_psf_block()
mol._update_atom_type(join(path, 'formate.str'))
df = mol.properties.atoms
df.loc[df['atom name'] == 'Cd', 'charge'] = 2.0
df.loc[df['atom name'] == 'Se', 'charge'] = -2.0
mol.as_psf(join(path, 'qd.psf'))

# Generate (CP2K) job settings
s = get_template('md_cp2k.yaml')
s.input.force_eval.subsys.topology = join(path, 'qd.psf')
s.input.force_eval.mm.forcefield.parm_file_name = join(path, 'par_all36_cgenff.prm')

# Set charges & LJ potential
lj_df = template_to_df('LJ_potential.yaml')
charge_df = template_to_df('atomic_charges.yaml')
set_lennard_jones(s, lj_df)
set_atomic_charges(s, charge_df)
set_subsys_kind(s, df)

# Generate a series of parameters
param = dict_to_pandas(get_template('param.yaml'), 'param')
param['key'] = ''
with pd.option_context('mode.chained_assignment', None):
    param.loc['charge'].update(charge_df)
    param.loc['epsilon'].update(lj_df)
    param.loc['sigma'].update(lj_df)

# Set charge constraints
charge_constrain = Settings()
charge_constrain.Cd = {'Se': -1, 'OG2D2': -0.5}
charge_constrain.Se = {'Cd': -1, 'OG2D2': 0.5}
charge_constrain.OG2D2 = {'Cd': -2, 'Se': 2}

# Prepare the ARMC settings
carlos = ARMC(param, mol)
carlos.job.path = path
carlos.job.charge_series = df['charge']
carlos.job.charge_series.index = df['atom type']
carlos.pes.rdf.func = MultiMolecule.init_rdf
carlos.pes.rdf.kwarg = {'atom_subset': ('Cd', 'Se', 'O')}
carlos.pes.rdf.ref = mol.init_rdf(**carlos.pes.rdf.kwarg)
carlos.job.settings = s
carlos.armc.iter_len = 100
carlos.armc.sub_iter_len = 10
