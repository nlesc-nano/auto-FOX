""" A module for calculating root mean squared (RMS) related properties. """

import numpy as np
import pandas as pd

from scm.plams.tools.periodic_table import PeriodicTable

from FOX.functions.read_xyz import read_multi_xyz
from FOX.examples.example_xyz import get_example_xyz


# Define constants
dr = 0.05
r_max = 12.0
atoms = ('Cd', 'Se')
xyz_file = get_example_xyz()

xyz_array, idx_dict = read_multi_xyz(xyz_file)
xyz_array2 = xyz_array.copy() + np.random.rand(xyz_array.shape[0])[:, None, None]

def get_rms(A, B):
    return np.linalg.norm(A - B)

def rotate_origin():
    pass


def translate_origin(xyz_array, idx_dict, atoms=None):
    """ Reset the origin by setting it to the center of mass. The center of mass can derived either
    from all atoms in the molecule (atoms=None) or a subset of atoms. """
    # Make sure we're dealing with a 3d array
    if len(xyz_array.shape) == 2:
        xyz_array = xyz_array[None, :, :]

    # If *atoms* is None: extract atomic symbols from they keys of *idx_dict*
    atoms = atoms or tuple(idx_dict.keys())

    # Create and fill the mass array
    mass_array = np.zeros(xyz_array.shape[1])
    for at in atoms:
        i = idx_dict[at]
        mass_array[i] = PeriodicTable.get_mass(at)
    mass_array = mass_array[None, :, None]

    # Create an array with centres of masses; subsequently translate xyz_array
    center_of_mass = np.sum(xyz_array * mass_array, axis=1) / mass_array.sum()
    return xyz_array - center_of_mass[:, None, :]


A = translate_origin(xyz_array, idx_dict, atoms=atoms)
A = rotate_origin(A)
B = translate_origin(xyz_array2, idx_dict, atoms=atoms)
B = rotate_origin(B)
rms = get_rms(A, B)
