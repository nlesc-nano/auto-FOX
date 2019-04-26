""" An input file example. """

__all__ = []

import time

from FOX import (MultiMolecule, get_example_xyz)


# Define the atoms of interest and the .xyz path + filename
atoms = ('Cd', 'Se', 'O')
example_xyz_filename = get_example_xyz()

# Optional: start the timer
print('')
start = time.time()

# Read the .xyz file
mol = MultiMolecule.from_xyz(example_xyz_filename)

# Calculate the RDF, RSMF & RMSD
rdf = mol.init_rdf(atom_subset=atoms)
# adf = mol.init_adf(atom_subset=atoms)  # Note: This is rather slow and can take a couple of minutes
rmsf = mol.init_rmsf(atom_subset=atoms)
rmsd = mol.init_rmsd(atom_subset=atoms)

# Optional: print the results and try to plot them in a graph (if Matplotlib is installed)
print('run time:', '%.2f' % (time.time() - start), 'sec')
try:
    rdf.plot()
    # adf.plot()
    rmsf.plot()
    rmsd.plot()
except Exception as ex:
    print(ex)
