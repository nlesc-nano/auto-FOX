""" A module for constructing radial distribution functions. """

__all__ = ['get_all_radial']

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def get_all_radial(xyz_array, idx_dict, dr=0.05, r_max=12.0, atoms=None):
    """ Return the radial distribution functions (RDFs) for all possible atom-pairs in **atoms**
    as dataframe. Accepts both 2d and 3d arrays of cartesian coordinates as input.

    :parameter xyz_array: A *m*n*3* or *n*3* numpy array with the cartesian coordinates of *m*
        molecules consisting of *n* atoms.
    :type xyz_array: np.ndarray_
    :parameter dict idx_dict: A dictionary consisting with atomic symbols as keys and matching
        atomic indices as values (|str|_: |list|_ [|int|_]).
    :parameter float dr: The integration step-size in Angstrom, *i.e.* the distance between
        concentric spheres.
    :parameter float r_max: The maximum to be evaluated interatomic distance.
    :parameter atoms: A tuple of atomic symbols. RDFs will be calculated for all
        possible atom-pairs in **atoms**. If *None*, calculate RDFs for all possible atom-pairs in
        the keys of **idx_dict** (*i.e.* all possible atoms pairs in the molecule).
    :type atoms: None or tuple [str]
    :return: A dataframe (|pd.DataFrame|_) of radial distribution functions, averaged over all
        conformations in **xyz_array**.
    """
    # Make sure we're dealing with a 3d array
    if len(xyz_array.shape) == 2:
        xyz_array = xyz_array[None, :, :]

    # If *atoms* is None: extract atomic symbols from they keys of *idx_dict*
    atoms = atoms or tuple(idx_dict.keys())

    # Construct a list of 2-tuples containing all unique atom pairs
    atom_pairs = [(at1, at2) for i, at1 in enumerate(atoms) for at2 in atoms[i:]]

    # Construct an empty dataframe with appropiate dimensions, indices and keys
    df = get_empty_df(dr, r_max, atom_pairs)

    # Fill the dataframe with RDF's, summed over all conformations in mol_list
    for xyz in xyz_array:
        for at1, at2 in atom_pairs:
            df[at1 + ' ' + at2] += get_radial_distr(xyz[idx_dict[at1]],
                                                    xyz[idx_dict[at2]],
                                                    dr=dr, r_max=r_max)

    # Remove np.nan and average the RDF's over all conformations in mol_list
    df.iloc[0] = 0.0
    df /= xyz_array.shape[0]
    return df


def get_empty_df(dr, r_max, atom_pairs):
    """ Construct and return a pandas dataframe filled with zeros.
    dr <float>: The stepsize.
    r_max: <float>: The maximum length.
    atom_pairs <list>: An list of 2-tuples representing the keys of the dataframe. """
    # Prepare the DataFrame arguments
    shape = 1 + int(r_max / dr), len(atom_pairs)
    index = np.arange(0, r_max + dr, dr)
    columns = [at1 + ' ' + at2 for at1, at2 in atom_pairs]

    # Create and return the DataFrame
    df = pd.DataFrame(np.zeros(shape), index=index, columns=columns)
    df.columns.name = 'Atom pairs'
    df.index.name = 'r  /  Ångström'
    return df


def get_radial_distr(array1, array2, dr=0.05, r_max=12.0):
    """ Calculate the radial distribution function between *array1* and *array2*: g(r_ij).

    array1 <np.ndarray>: A n*3 array of the cartesian coordinates of reference atoms.
    array2 <np.ndarray>: A m*3 array of the cartesian coordinates of (non-reference) atoms.
    dr <float>: The integration step-size in Angstrom, i.e. the distance between concentric spheres.
    r_max <float>: The maximum to be evaluated interatomic distance.
    return <np.ndarray>: The radial distribution function: a 1d array of length *dr* / *r_max*.
    """
    idx_max = 1 + int(r_max / dr)
    dist = cdist(array1, array2)
    dist_int = np.array(dist / dr, dtype=int).flatten()

    # Calculate the average particle density N / V
    # The diameter of the spherical volume (V) is defined by the largest inter-particle distance
    dens_mean = len(array2) / ((4/3) * np.pi * (0.5 * dist.max())**3)

    # Count the number of occurances of each (rounded) distance
    dens = np.bincount(dist_int)[:idx_max]

    # Correct for the number of reference atoms
    dens = dens / len(array1)
    dens[0] = np.nan

    # Convert the particle count into a partical density
    r = np.arange(0, r_max + dr, dr)
    try:
        dens /= (4 * np.pi * r**2 * dr)
    except ValueError:
        # Plan b: Pad the array with zeros if r_max is larger than dist.max()
        zeros = np.zeros(len(r))
        zeros[0:len(dens)] = dens
        dens = zeros / (4 * np.pi * r**2 * dr)

    # Normalize and return the particle density
    return dens / dens_mean
