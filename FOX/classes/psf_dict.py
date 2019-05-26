"""A class for reading protein structure (.psf) files."""

from __future__ import annotations

from itertools import chain

import numpy as np
import pandas as pd
from typing import Dict

from collections import defaultdict

from FOX.classes.multi_mol import MultiMolecule
from FOX.functions.utils import (serialize_array, read_str_file)

__all__ = ['PSFDict']


class PSFDict(defaultdict):
    """A class for storing protein structure files.

    Keys within this class are frozen (they cannot be added, altered or removed).
    The available name of keys and type+shape, the matching values and their meanings are displayed
    below.

    :value filename: An array with a single string as filename.
    :type filename: :math:`1*1` |np.ndarray|_ [|np.str_|_]
    :value title: An array of strings holding the *title* block.
    :type title: :math:`g*1` |np.ndarray|_ [|np.str_|_]
    :value atoms: A Pandas DataFrame holding the *atoms* block.
    :type atoms: :math:`h*8` |pd.DataFrame|_
    :value bonds: An array holding the indices of all atom-pairs defining bonds.
    :type bonds: :math:`i*2` |np.ndarray|_ [|np.int64|_]
    :value angles: An array holding the indices of all atoms defining angles.
    :type angles: :math:`j*3` |np.ndarray|_ [|np.int64|_]
    :value dihedrals: An array holding the indices of all atoms defining proper
        dihedral angles.
    :type dihedrals: :math:`k*4` |np.ndarray|_ [|np.int64|_]
    :value impropers: An array holding the indices of all atoms defining improper
        dihedral angles.
    :type impropers: :math:`l*4` |np.ndarray|_ [|np.int64|_]
    :value donors: An array holding the atomic indices of all hydrogen-bond donors.
    :type donors: :math:`m*1` |np.ndarray|_ [|np.int64|_]
    :value acceptors: An array holding the atomic indices of all hydrogen-bond acceptors.
    :type acceptors: :math:`n*1` |np.ndarray|_ [|np.int64|_]
    :value no_nonbonded: An array holding the indices of all atom-pairs whose nonbonded
        interactions should be ignored.
    :type no_nonbonded: :math:`o*2` |np.ndarray|_ [|np.int64|_]
    """

    def __init__(self, **kwarg: Dict[str, np.ndarray]) -> None:
        _key_dict = {
            'filename': {'shape': 1},
            'title': {'shape': 1},
            'atoms': {'shape': 8},
            'bonds': {'shape': 2, 'row_len': 4, 'header': '{:>10d} !NBOND: bonds'},
            'angles': {'shape': 3, 'row_len': 3, 'header': '{:>10d} !NTHETA: angles'},
            'dihedrals': {'shape': 4, 'row_len': 2, 'header': '{:>10d} !NPHI: dihedrals'},
            'impropers': {'shape': 4, 'row_len': 2, 'header': '{:>10d} !NIMPHI: impropers'},
            'donors': {'shape': 1, 'row_len': 8, 'header': '{:>10d} !NDON: donors'},
            'acceptors': {'shape': 1, 'row_len': 8, 'header': '{:>10d} !NACC: acceptors'},
            'no_nonbonded': {'shape': 1, 'row_len': 4, 'header': '{:>10d} !NNB'}
        }

        for key in kwarg:
            if key not in _key_dict:
                raise KeyError(key)
        defaultdict.__init__(self, **kwarg)
        defaultdict.__setitem__(self, '_key_dict', _key_dict)
        for key in _key_dict:
            defaultdict.setdefault(self, key)

    def __missing__(self, name: str) -> None:
        return None

    def __setitem__(self, name: str, value: np.ndarray) -> None:
        if name not in self['_key_dict']:
            err = 'Invalid key: {}. Valid keys are {}'
            raise ValueError(err.format(str(name), str(self._key_dict)))
        shape = self['_key_dict'][name]['shape']

        if value is None:
            defaultdict.__setitem__(self, name, None)
        elif name == 'filename' and isinstance(value, str):
            defaultdict.__setitem__(self, name, np.array(value, ndmin=1))
        elif value.ndim == 1 and shape == 1:
            defaultdict.__setitem__(self, name, value[:, None])
        elif value.ndim == 2 and value.shape[1] == shape:
            defaultdict.__setitem__(self, name, value)
        else:
            dim = '*'.join(['{:d}'.format(i) for i in value.shape])
            err = "Value has an invalid shape: {}. '{}' requires a n*{:d} array"
            raise ValueError(err.format(dim, name, shape))

    def __delitem__(self, name: str) -> None:
        if name == '_type_dict':
            raise ValueError("'_type_dict' cannot be deleted")
        self[name] = None

    def __getattr__(self, name: str) -> any:
        if (name.startswith('__') and name.endswith('__')):
            return defaultdict.__getattribute__(self, name)
        return self[name]

    def __setattr__(self, name: str, value: np.ndarray) -> None:
        if name.startswith('__') and name.endswith('__'):
            defaultdict.__setattr__(self, name, value)
        self[name] = value

    def __delattr__(self, name: str) -> None:
        if name.startswith('__') and name.endswith('__'):
            defaultdict.__delattr__(self, name)
        del self[name]

    def __str__(self) -> str:
        ret = ''
        item = '\tobject:\t {}\n\tshape:\t {}'
        for key, value in self.items():
            if key == '_type_dict':
                continue
            ret += item.format(str(type(value)),
                               str(value.shape)
                               )
        return ret

    def __repr__(self) -> str:
        return _str(self)

    def as_dict(self) -> Dict[str, np.ndarray]:
        """Convert a :class:`.PSFDict` instance into a dictionary."""
        return {key: value for key, value in self.items()}

    def set_filename(self, filename: str) -> None:
        """Set the filename of a .psf file.

        The value is used by the :meth:`write_psf` method, serving as the filename of a to-be
        created .psf file.

        :parameter str filename: The path+filename of a to-be created .psf file.
        """
        self.filename = np.array(filename, ndmin=1)

    @classmethod
    def read_psf(cls, filename: str) -> PSFDict:
        """Read a protein structure file (.psf) and return the various .psf blocks as a dictionary.

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

        Parameters
        ----------
        str filename: The path + filename of a .psf file.

        Returns
        -------
        |dict|_ (keys: |str|_):
            A dictionary holding the content of a .psf file.
            
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

        ret: dict = {}
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

        return cls(**PSFDict._post_process_psf(ret))

    @staticmethod
    def _post_process_psf(psf_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Post-process the output of :func:`.read_psf`, casting the values into appropiate
        objects:

        * The title block is converted into an (un-nested) list of strings.
        * The atoms block is converted into a Pandas DataFrame.
        * All other blocks are converted into a flattened array of integers.

        Parameters
        ----------
        dict psf_dict:
            A dictionary holding the content of a .psf file (see :func:`.read_psf`).

        Returns
        -------
        |dict|_ (keys: |str|_):
            The .psf output, **psf_dict**, with properly formatted values.

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
                psf_dict[key] = np.array([' '.join(j for j in i).strip('REMARKS ') for i in value])

        return psf_dict

    def write_psf(self, filename: str = None) -> None:
        """Create a protein structure file (.psf) out of a :class:`.PSFDict` instance.

        Parameters
        ----------
        str filename:
            The path+filename of the .psf file.
            If ``None``, try to pull the name from **self['filename']**
            (see :meth:`set_filename`).
            
        """
        try:
            filename = filename or self.filename[0]
        except TypeError:
            raise TypeError("'filename' is missing")

        # Prepare the !NTITLE block
        top = 'PSF EXT\n'
        if self.title is None:
            top += '\n{:>10d} !NTITLE'.format(2)
            top += '\n{:>10.10} PSF file generated with Auto-FOX:'.format('REMARKS')
            top += '\n{:>10.10} https://github.com/nlesc-nano/auto-FOX'.format('REMARKS')
        else:
            top += '\n{:>10d} !NTITLE'.format(self.title.shape[0])
            for i in self.title:
                top += '\n{:>10.10} '.format('REMARKS') + i

        # Prepare the !NATOM block
        if self.atoms is not None:
            top += '\n\n{:>10d} !NATOM\n'.format(self.atoms.shape[0])
            string = '{:>10d} {:8.8} {:<8d} {:8.8} {:8.8} {:6.6} {:>9f} {:>15f} {:>8d}'
            for i, j in self.atoms.iterrows():
                top += string.format(*[i]+j.values.tolist()) + '\n'
        else:
            top += '\n\n{:>10d} !NATOM\n'.format(0)

        # Prepare the !NBOND, !NTHETA, !NPHI, !NIMPHI, !NDON, !NACC and !NNB blocks
        bottom = ''
        sections = ('bonds', 'angles', 'dihedrals', 'impropers',
                    'donors', 'acceptors', 'no_nonbonded')
        for key in sections:
            value = self[key]
            if key in ('title', 'atoms', '_key_dict'):
                continue
            header = self._key_dict[key]['header']
            row_len = self._key_dict[key]['row_len']
            if value is None:
                bottom += '\n\n' + header.format(0)
            else:
                bottom += '\n\n' + header.format(value.shape[0])
                bottom += '\n' + serialize_array(value, row_len)

        # Write the .psf file
        with open(filename, 'w') as f:
            f.write(top)
            f.write(bottom[1:])

    @classmethod
    def from_multi_mol(cls, multi_mol: MultiMolecule) -> PSFDict:
        """Construct :class:`PSFdict` instance from a :class:`.MultiMolecule` instance."""
        res = multi_mol.residue_argsort(concatenate=False)
        plams_mol = multi_mol.as_Molecule(0)[0]
        plams_mol.fix_bond_orders()

        # Construct the .psf dataframe
        df = pd.DataFrame(index=np.arange(1, multi_mol.shape[1]+1))
        df.index.name = 'ID'
        df['segment name'] = 'MOL'
        df['residue ID'] = [i for i, j in enumerate(res, 1) for _ in j]
        df['residue name'] = ['COR' if i == 1 else 'LIG' for i in df['residue ID']]
        df['atom name'] = multi_mol.symbol
        df['atom type'] = df['atom name']
        df['charge'] = [at.properties.charge for at in plams_mol]
        df['mass'] = multi_mol.mass
        df['0'] = 0

        # Prepare arguments for constructing the 'segment name' column
        key = sorted(set(df.loc[df['residue ID'] == 1, 'atom type']))
        value = range(1, len(key) + 1)
        segment_dict = dict(zip(key, value))
        value_max = 'MOL' + str(value.stop)

        # Construct the 'segment name' column
        segment_name = []
        for item in df['atom name']:
            try:
                segment_name.append('MOL{:d}'.format(segment_dict[item]))
            except KeyError:
                segment_name.append(value_max)
        df['segment name'] = segment_name

        ret = {
            'atoms': df,
            'bonds': multi_mol.atom12 + 1,
            'angles': plams_mol.get_angles(),
            'dihedrals': plams_mol.get_dihedrals(),
            'impropers': plams_mol.get_impropers()
        }

        return cls(**ret)

    def update_atom_type(self,
                         filename: str) -> None:
        """Update all ligand atom types and atomic charges in self['atoms']."""
        at_type, charge = read_str_file(filename)
        id_range = range(2, max(self.atoms['residue ID'])+1)
        for i in id_range:
            j = self.atoms[self.atoms['residue ID'] == i].index
            self.atoms.loc[j, 'atom type'] = at_type
            self.atoms.loc[j, 'charge'] = charge

    def update_atom_charge(self, atom_type: str,
                           charge: float) -> None:
        """Change the charge of a specific atom type to **charge**."""
        self.atoms.loc[self.atoms['atom type'] == atom_type, 'charge'] = charge


def _str(dict_: dict,
         indent: str = '') -> str:
    ret = ''
    for key, value in dict_.items():
        ret += indent + str(key) + ': \t'
        if isinstance(value, dict):
            ret += '\n' + _str(value, indent+'\t')
        else:
            ret += str(value) + '\n'
    return ret
