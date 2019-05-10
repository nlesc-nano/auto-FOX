""" A module for reading protein structure (.psf) files. """

__all__ = ['read_psf', 'write_psf']

from itertools import chain

import numpy as np
import pandas as pd

from ..functions.utils import serialize_array


def read_psf(filename):
    """ Read a protein structure file (.psf) and return the various .psf blocks as a dictionary.

    Depending on the content of the .psf file, the dictionary can contain
    the following keys and values:

        * *title*: list of remarks (str)
        * *atoms*: A dataframe of atoms
        * *bonds*: A :math:`i*2` array of atomic indices defining bonds
        * *angles*: A :math:`j*3` array of atomic indices defining angles
        * *dihedrals*: A :math:`k*4` array of atomic indices defining proper dihedral angles
        * *impropers*: A :math:`l*4` array of atomic indices defining improper dihedral angles
        * *donors*: A :math:`m*1` array of atomic indices defining hydrogen-bond donors
        * *acceptors*: A :math:`n*1` array of atomic indices defining hydrogen-bond acceptors
        * *no_nonbonded*: A :math:`o*2` array of atomic indices defining to-be ignore nonbonded
          interactions

    The dictionary produced by this function be fed into :func:`.write_psf` to create a new .psf
    file:

    .. code:: python

        >>> psf_dict = read_psf('old_file.psf')
        >>> write_psf('new_file.psf', **psf_dict)

    :parameter str filename: The path + filename of a .psf file.
    :return: A dictionary holding the content of a .psf file.
    :rtype: |dict|_ (keys: |str|_)
    """
    header_dict = {'!NTITLE': 'title',
                   '!NATOM': 'atoms',
                   '!NBOND': 'bonds',
                   '!NTHETA': 'angles',
                   '!NPHI': 'dihedrals',
                   '!NIMPHI': 'impropers',
                   '!NDON': 'donors',
                   '!NACC': 'acceptors',
                   '!NNB': 'no_nonbonded'}

    ret = {}
    with open(filename, 'r') as f:
        next(f)  # Skip the first line
        for i in f:
            # Search for .psf blocks
            if i == '\n':
                continue

            # Read the .psf block header
            key = header_dict[i.split()[1].rstrip(':')]
            ret[key] = []

            # Read .psf blocks
            try:
                j = next(f)
            except StopIteration:
                break
            while j != '\n':
                ret[key].append(j.split())
                j = next(f)

    return _post_process_psf(ret)


def _post_process_psf(psf_dict):
    """ Post-process the output of :func:`.read_psf`, casting the values into appropiate
    objects:

    * The title block is converted into an (un-nested) list of strings.
    * The atoms block is converted into a Pandas DataFrame.
    * All other blocks are converted into a flattened array of integers.

    :parameter psf_dict: A dictionary holding the content of a .psf file (see :func:`.read_psf`).
    :type psf_dict: |dict|_ (keys: |str|_, values: |list|_[|list|_[|str|_]])
    :return: The .psf output, **psf_dict**, with properly formatted values.
    :rtype: |dict|_ (keys: |str|_)
    """
    shape_dict = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4,
                  'donors': 1,
                  'acceptors': 1,
                  'no_nonbonded': 2}

    for key, value in psf_dict.items():  # Post-process the output
        # Cast into a flattened array of indices
        if key not in ('title', 'atoms'):
            ar = np.fromiter(chain.from_iterable(value), dtype=int)
            ar.shape = len(ar) // shape_dict[key], shape_dict[key]
            psf_dict[key] = ar

        # Cast the atoms block into a dataframe
        elif key == 'atoms':
            df = pd.DataFrame(value)
            df[0] = df[0].astype(int, copy=False)
            df.set_index(0, inplace=True)
            df.index.name = 'ID'
            df.columns = ['segment name', 'residue ID', 'residue name',
                          'atom name', 'atom type', 'charge', 'mass', '0']
            df['residue ID'] = df['residue ID'].astype(int, copy=False)
            df['charge'] = df['charge'].astype(float, copy=False)
            df['mass'] = df['mass'].astype(float, copy=False)
            df['0'] = df['0'].astype(int, copy=False)
            psf_dict[key] = df

        # Cast the title in a list of strings
        elif key == 'title':
            psf_dict[key] = [' '.join(j for j in i).strip('REMARKS ') for i in value]

    return psf_dict


def write_psf(filename='mol.psf', title=None, atoms=None, bonds=None, angles=None, dihedrals=None,
              impropers=None, donors=None, acceptors=None, no_nonbonded=None):
    """ Create a protein structure file (.psf).

    :parameter title: A list of strings holding the *title* block
    :type title: |list|_ [|str|_]
    :parameter atoms: A Pandas DataFrame holding the *atoms* block.
    :type atoms: |pd.DataFrame|_
    :parameter bonds: An array holding the indices of all atom-pairs defining bonds.
    :type bonds: :math:`i*2` |np.ndarray|_ [|np.int64|_]
    :parameter angles: An array holding the indices of all atoms defining angles.
    :type angles: :math:`j*3` |np.ndarray|_ [|np.int64|_]
    :parameter dihedrals: An array holding the indices of all atoms defining proper
        dihedral angles.
    :type dihedrals: :math:`k*4` |np.ndarray|_ [|np.int64|_]
    :parameter impropers: An array holding the indices of all atoms defining improper
        dihedral angles.
    :type impropers: :math:`l*4` |np.ndarray|_ [|np.int64|_]
    :parameter donors: An array holding the atomic indices of all hydrogen-bond donors.
    :type donors: :math:`m*1` |np.ndarray|_ [|np.int64|_]
    :parameter acceptors: An array holding the atomic indices of all hydrogen-bond acceptors.
    :type acceptors: :math:`n*1` |np.ndarray|_ [|np.int64|_]
    :parameter no_nonbonded: An array holding the indices of all atom-pairs whose nonbonded
        interactions should be ignored.
    :type no_nonbonded: :math:`o*2` |np.ndarray|_ [|np.int64|_]
    """
    # Prepare the !NTITLE block
    top = 'PSF EXT\n'
    if title is None:
        top += '\n{:>10d} !NTITLE'.format(2)
        top += '\n{:>10.10} PSF file generated with Auto-FOX:'.format('REMARKS')
        top += '\n{:>10.10} https://github.com/nlesc-nano/auto-FOX'.format('REMARKS')
    else:
        top += '\n{:>10d} !NTITLE'.format(len(title))
        for i in title:
            top += '\n{:>10.10} '.format('REMARKS') + i

    # Prepare the !NATOM block
    if atoms is not None:
        top += '\n\n{:>10d} !NATOM\n'.format(atoms.shape[0])
        string = '{:>10d} {:8.8} {:<8d} {:8.8} {:8.8} {:6.6} {:>9f} {:>15f} {:>8d}'
        for i, j in atoms.iterrows():
            top += string.format(*[i]+j.values.tolist()) + '\n'
    else:
        top += '\n\n{:>10d} !NATOM\n'.format(0)

    # Prepare arguments
    row_len = [4, 3, 2, 2, 8, 8, 4]
    bottom_headers = {
        '{:>10d} !NBOND: bonds': bonds,
        '{:>10d} !NTHETA: angles': angles,
        '{:>10d} !NPHI: dihedrals': dihedrals,
        '{:>10d} !NIMPHI: impropers': impropers,
        '{:>10d} !NDON: donors': donors,
        '{:>10d} !NACC: acceptors': acceptors,
        '{:>10d} !NNB': no_nonbonded
    }

    # Prepare the !NBOND, !NTHETA, !NPHI, !NIMPHI, !NDON, !NACC and !NNB blocks
    bottom = ''
    for i, (key, value) in zip(row_len, bottom_headers.items()):
        if value is None:
            bottom += '\n\n' + key.format(0)
        else:
            bottom += '\n\n' + key.format(value.shape[0])
            bottom += '\n' + serialize_array(value, i)

    # Write the .psf file
    with open(filename, 'w') as f:
        f.write(top)
        f.write(bottom[1:])
