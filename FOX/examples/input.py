""" An input file example. """

__all__ = []

import time

from FOX.functions.read_xyz import read_multi_xyz
from FOX.functions.radial_distribution import get_all_radial
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
xyz_array, idx_dict = read_multi_xyz(xyz_file)
df = get_all_radial(xyz_array, idx_dict, dr=dr, r_max=r_max, atoms=atoms)

# Print the results
print('run time:', '%.2f' % (time.time() - start), 'sec')
try:
    df.plot()
except Exception as ex:
    print(ex)
