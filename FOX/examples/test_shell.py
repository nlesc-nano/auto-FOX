from FOX.classes.multi_mol import MultiMolecule
from FOX.examples.example_xyz import get_example_xyz
import matplotlib.pyplot as plt

example_xyz_file = get_example_xyz()
mol = MultiMolecule(filename=example_xyz_file)
rmsf, rmsf_idx, rdf = mol.init_shell_search(atom_subset=('Cd', 'Se'))

fig, (ax, ax2) = plt.subplots(ncols=2)
rmsf.plot(ax=ax)
rdf.plot(ax=ax2)
plt.show()