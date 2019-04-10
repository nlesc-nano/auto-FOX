""" A module for constructing angular distribution functions. """

__all__ = ['get_adf']

import numpy as np
import pandas as pd


def get_adf_df(atom_pairs):
    """ Construct and return a pandas dataframe filled with zeros to hold angular distribution
    functions.

    :parameter atom_pairs: An list of 3-tuples representing the keys of the dataframe.
    :type atom_pairs: |list|_ [|tuple|_ [|str|_]]
    :return: An empty dataframe.
    :rtype: |pd.DataFrame|_
    """
    # Prepare the DataFrame arguments
    shape = 181, len(atom_pairs)
    index = np.arange(181)
    try:  # If **atom_pairs** consists of atomic symbols
        columns = [at1 + ' ' + at2 + ' ' + at3 for at1, at2, at3 in atom_pairs]
    except TypeError:  # If **atom_pairs** consists of atomic indices
        columns = ['series ' + str(i) for i, _ in enumerate(atom_pairs, 1)]

    # Create and return the DataFrame
    df = pd.DataFrame(np.zeros(shape), index=index, columns=columns)
    df.columns.name = 'Atom pairs'
    df.index.name = 'phi  /  Degrees'
    return df


def get_adf(ang_mat, r_max=24.0):
    """ Calculate and return the radial distribution function (RDF) based on the 2D distance matrix
    **dist**. A more memory efficient implementation of :func:`FOX.functions.rdf.get_rdf`,
    which operates on a 3D dstance matrix.

    :parameter ang: A 4D angle matrix constructed from *m* molecules and three sets of *n*, *k* and
        *l* atoms.
    :type ang: *m*n*k*l* |np.ndarray|_ [|np.float64|_]
    :parameter float r_max: The diameter of the sphere used for converting particle counts into
        densities.
    :return: A 1D array with an angular distribution function spanning all values between 0 and 180
        degrees.
    :rtype: 181 |np.ndarray|_ [|np.float64|_]
    """
    ang_mat[np.isnan(ang_mat)] = 10
    ang_int = np.array(np.degrees(ang_mat), dtype=int)
    volume = (4/3) * np.pi * (0.5 * r_max)**3
    dens_mean = ang_mat.shape[1] / volume

    dens = np.array([np.bincount(i.flatten(), minlength=181)[:181] for i in ang_int], dtype=float)
    dens /= np.product(ang_mat.shape[-2:])  # Correct for the number of reference atoms
    dens /= volume / 181  # Convert the particle count into a partical density
    dens /= dens_mean  # Normalize the particle density
    return np.average(dens, axis=0)
