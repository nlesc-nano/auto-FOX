"""An input file example."""

import time

from FOX import (MultiMolecule, get_example_xyz)

__all__ = []

rdf = adf = rmsf = rmsd = None

# Define the atoms of interest and the .xyz path + filename
atoms = ('Cd', 'Se', 'O')
example_xyz_filename = get_example_xyz()

# Optional: start the timer
start = time.time()

# Read the .xyz file
mol = MultiMolecule.from_xyz(example_xyz_filename)

# Calculate the RDF, RSMF & RMSD
rdf = mol.init_rdf(atom_subset=atoms)
adf = mol.init_adf(r_max=8.0, atom_subset=['Cd', 'Se'])
rmsf = mol.init_rmsf(atom_subset=atoms)
rmsd = mol.init_rmsd(atom_subset=atoms)

# Optional: print the results and try to plot them in a graph (if Matplotlib is installed)
for name, df in {'rdf ': rdf, 'adf ': adf, 'rmsf': rmsf, 'rmsd': rmsd}.items():
    try:
        df.plot(name)
    except Exception as ex:
        print(f'{name} - {ex.__class__.__name__}: {ex}')
print('run time: {:.2f} sec'.format(time.time() - start))
