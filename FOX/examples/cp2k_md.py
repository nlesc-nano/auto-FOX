""" A work in progress recipe for MM-MD parameter optimizations with CP2K. """

from os.path import join

from scm.plams import add_to_class

from FOX.classes.multi_mol import MultiMolecule
from FOX.examples.example_xyz import get_example_xyz
from FOX.functions.utils import get_template
from FOX.functions.cp2k_utils import set_subsys_kind, set_lennard_jones


@add_to_class(Cp2kJob)
def get_runscript(self):
    return 'mpirun cp2k.popt -i {} -o {}'.format(self._filename('inp'), self._filename('out'))


# Read the .xyz file
path = '/Users/basvanbeek/Downloads'
mol = MultiMolecule(filename=get_example_xyz())
mol.guess_bonds(atom_subset=['C', 'O', 'H'])
mol._set_psf_block()
mol._update_atom_type(join(path, 'formate.str'))
df = mol.properties.atoms
df.loc[df['atom name'] == 'Cd', 'charge'] = 2
df.loc[df['atom name'] == 'Se', 'charge'] = -2
mol.as_psf(join(path, 'qd.psf'))

s = get_template('md_cp2k.yaml')
lj_dict = get_template('LJ_potential.yaml')

s.input.force_eval.subsys.topology = join(path, 'qd.psf')
s.input.force_eval.mm.forcefield.parm_file_name = join(path, 'par_all36_cgenff.prm')
set_subsys_kind(s, df)
set_lennard_jones(s, lj_dict)
