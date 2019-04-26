""" A Recipe for MM-MD parameter optimizations with CP2K 3.1 or 6.1. """

from scm.plams import Settings

from FOX import (ARMC, MultiMolecule)
from FOX.functions.utils import (get_template, dict_to_pandas, get_example_xyz)
from FOX.functions.cp2k_utils import (set_subsys_kind, set_keys)


# Read the .xyz file
mol = MultiMolecule.from_xyz(get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol.generate_psf_block()
mol.update_atom_type('formate.str')
df = mol.properties.psf
df.loc[df['atom name'] == 'Cd', 'charge'] = 2.0
df.loc[df['atom name'] == 'Se', 'charge'] = -2.0

# Generate (CP2K) job settings and parameters
s = get_template('md_cp2k.yaml')
s.input.force_eval.subsys.topology.conn_file_name = 'qd.psf'
s.input.force_eval.mm.forcefield.parm_file_name = 'par_all36_cgenff.prm'
param = dict_to_pandas(get_template('param.yaml'), 'param')
set_subsys_kind(s, df)
set_keys(s, param)

# Set charge constraints
charge_constrain = Settings()
charge_constrain.Cd = {'Se': -1, 'OG2D2': -0.5}
charge_constrain.Se = {'Cd': -1, 'OG2D2': 0.5}
charge_constrain.OG2D2 = {'Cd': -2, 'Se': 2}

# Prepare the ARMC settings
armc = ARMC(param, mol)
armc.job.settings = s
armc.job.psf = mol.as_psf('qd.psf', return_blocks=True)
armc.pes.rdf.func = MultiMolecule.init_rdf
armc.pes.rdf.kwarg = {'atom_subset': ('Cd', 'Se', 'O')}
armc.pes.rdf.ref = mol.init_rdf(**armc.pes.rdf.kwarg)

# Start the MC parameterization
armc.init_armc()
