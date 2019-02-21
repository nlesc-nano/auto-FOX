""" An input file example. """

__all__ = []

import time

from FOX.functions.multi_mol import MultiMolecule
from FOX.examples.example_xyz import get_example_xyz


# Define constants
dr = 0.05
r_max = 12.0
atoms = ('Cd', 'Se', 'O')
xyz_file = get_example_xyz()

# Start the timer
print('')
start = time.time()

# Run the actual script
mol = MultiMolecule(filename=xyz_file)
df = mol.init_rdf(dr=dr, r_max=r_max, atom_subset=atoms)

# Print the results
print('run time:', '%.2f' % (time.time() - start), 'sec')
try:
    df.plot()
except Exception as ex:
    print(ex)
