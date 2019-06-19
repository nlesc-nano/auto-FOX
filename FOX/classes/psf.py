"""A class for reading protein structure (.psf) files."""

from __future__ import annotations

from typing import (Dict, Optional)
from dataclasses import (dataclass, field)

import numpy as np
import pandas as pd

from .multi_mol import MultiMolecule
from .frozen_settings import FrozenSettings
from ..functions.utils import (serialize_array, read_str_file)

__all__ = ['PSF']


@dataclass
class PSF:
    """A dataclass for storing and parsing protein structure files.

    Attributes
    ----------
    filename : :math:`1*1` |np.ndarray|_ [|np.str_|_]
        An array with a single string as filename.

    title : :math:`g*1` |np.ndarray|_ [|np.str_|_]
        An array of strings holding the ``"title"`` block.

    atoms : :math:`h*8` |pd.DataFrame|_
        A Pandas DataFrame holding the ``"atoms"`` block.

    bonds : :math:`i*2` |np.ndarray|_ [|np.int64|_]
        An array holding the indices of all atom-pairs defining bonds.

    angles : :math:`j*3` |np.ndarray|_ [|np.int64|_]
        An array holding the indices of all atoms defining angles.

    dihedrals : :math:`k*4` |np.ndarray|_ [|np.int64|_]
        An array holding the indices of all atoms defining proper dihedral angles.

    impropers : :math:`l*4` |np.ndarray|_ [|np.int64|_]
        An array holding the indices of all atoms defining improper dihedral angles.

    donors : :math:`m*1` |np.ndarray|_ [|np.int64|_]
        An array holding the atomic indices of all hydrogen-bond donors.

    acceptors : :math:`n*1` |np.ndarray|_ [|np.int64|_]
        An array holding the atomic indices of all hydrogen-bond acceptors.

    no_nonbonded : :math:`o*2` |np.ndarray|_ [|np.int64|_]
        An array holding the indices of all atom-pairs whose nonbonded
        interactions should be ignored.

    _key_dict : |FrozenSettings|_
        An imutable :class:`.FrozenSettings` instance holding array shapes, row lengths and headers.

    """
    filename: Optional[np.ndarray] = None
    title: Optional[np.ndarray] = None
    atoms: Optional[pd.DataFrame] = None
    bonds: Optional[np.ndarray] = None
    angles: Optional[np.ndarray] = None
    dihedrals: Optional[np.ndarray] = None
    impropers: Optional[np.ndarray] = None
    donors: Optional[np.ndarray] = None
    acceptors: Optional[np.ndarray] = None
    no_nonbonded: Optional[np.ndarray] = None
    _key_dict: FrozenSettings = field(init=False)

    def __post_init__(self) -> None:
        self._key_dict = FrozenSettings({
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
        })

    def set_filename(self, filename: str) -> None:
        """Set the filename of a .psf file.

        The value is used by the :meth:`write_psf` method, serving as the filename of a to-be
        created .psf file. Performs in inplace update of :attr:`.PSF.filename`

        Parameters
        ----------
        filename : str
            The path+filename of the .psf file.

        """
        self.filename = np.array(filename, ndmin=1)

    @classmethod
    def read_psf(cls, filename: str) -> PSF:
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

        Examples
        --------
        The dictionary produced by this function be fed into :func:`.write_psf` to create a new .psf
        file:

        .. code:: python

            >>> psf_dict = read_psf('old_file.psf')
            >>> write_psf('new_file.psf', **psf_dict)

        Parameters
        ----------
        str filename:
            The path + filename of a .psf file.

        Returns
        -------
        |FOX.PSF|_:
            A :class:`.PSF` instance holding the content of a .psf file.

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
                ret[key] = value = []

                # Read .psf blocks
                j = next(f)
                while j != '\n':
                    value.append(j.split())
                    try:
                        j = next(f)
                    except StopIteration:
                        break

        return cls(**PSF._post_process_psf(ret))

    def _post_process_psf(self, psf_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Post-process the output of :func:`.read_psf`, casting the values into appropiate
        objects:

        * The title block is converted into an (un-nested) list of strings.
        * The atoms block is converted into a Pandas DataFrame.
        * All other blocks are converted into a flattened array of integers.

        Parameters
        ----------
        psf_dict : dict [str, |np.ndarray|_]
            A dictionary holding the content of a .psf file (see :func:`.read_psf`).

        Returns
        -------
        |dict|_ [|str|_, |np.ndarray|_]:
            The .psf output, **psf_dict**, with properly formatted values.

        """
        for key, value in psf_dict.items():  # Post-process the output
            # Cast into a flattened array of indices
            if key not in ('title', 'atoms'):
                ar = np.array(value, dtype=int).ravel()
                ar.shape = len(ar) // self._shape_dict[key].row_len, self._shape_dict[key].row_len
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
                psf_dict[key] = np.array([' '.join(i).strip('REMARKS ') for i in value])

        return psf_dict

    def write_psf(self, filename: Optional[str] = None) -> None:
        """Create a protein structure file (.psf) out of a :class:`.PSF` instance.

        Parameters
        ----------
        filename : str
            The path+filename of the .psf file.
            If ``None``, try to pull the name from :attr:`.PSF.filename`
            (see :meth:`.PSF.set_filename`).

        Raises
        ------
        TypeError
            Raised if the filename is specified in neither **filename** nor :attr:`.PSF.filename`.

        """
        try:
            filename = filename or self.filename[0]
        except TypeError:
            raise TypeError("The 'filename' argument is missing")

        top = self._parse_top()
        bottom = self._parse_bottom()

        # Write the .psf file
        with open(filename, 'w') as f:
            f.write(top)
            f.write(bottom[1:])

    def _serialize_top(self) -> str:
        """Serialize the top-most section of the to-be create .psf file.

        The following blocks are seralized:

            * :attr:`.PSF.title`
            * :attr:`.PSF.atoms`

        Returns
        -------
        |str|_
            A string constructed from the above-mentioned psf blocks.

        See Also
        --------
        :meth:`.PSF.write_psf`
            The main method for writing .psf files.

        """
        # Prepare the !NTITLE block
        ret = 'PSF EXT\n'
        if self.title is None:
            ret += '\n{:>10d} !NTITLE'.format(2)
            ret += '\n{:>10.10} PSF file generated with Auto-FOX:'.format('REMARKS')
            ret += '\n{:>10.10} https://github.com/nlesc-nano/auto-FOX'.format('REMARKS')
        else:
            ret += '\n{:>10d} !NTITLE'.format(self.title.shape[0])
            for i in self.title:
                ret += '\n{:>10.10} '.format('REMARKS') + i

        # Prepare the !NATOM block
        if self.atoms is not None:
            ret += '\n\n{:>10d} !NATOM\n'.format(self.atoms.shape[0])
            string = '{:>10d} {:8.8} {:<8d} {:8.8} {:8.8} {:6.6} {:>9f} {:>15f} {:>8d}'
            for i, j in self.atoms.iterrows():
                ret += string.format(*[i]+j.values.tolist()) + '\n'
        else:
            ret += '\n\n{:>10d} !NATOM\n'.format(0)
        return ret

    def _serialize_bottom(self) -> str:
        """Serialize the bottom-most section of the to-be create .psf file.

        The following blocks are seralized:

            * :attr:`.PSF.bonds`
            * :attr:`.PSF.angles`
            * :attr:`.PSF.dihedrals`
            * :attr:`.PSF.impropers`
            * :attr:`.PSF.donors`
            * :attr:`.PSF.acceptors`
            * :attr:`.PSF.no_nonbonded`

        Returns
        -------
        |str|_
            A string constructed from the above-mentioned psf blocks.

        See Also
        --------
        :meth:`.PSF.write_psf`
            The main method for writing .psf files.

        """
        sections = ('bonds', 'angles', 'dihedrals', 'impropers',
                    'donors', 'acceptors', 'no_nonbonded')

        ret = ''
        for attr in sections:
            if attr in ('title', 'atoms', '_key_dict'):
                continue

            header = self._key_dict[attr].header
            row_len = self._key_dict[attr].row_len
            value = getattr(self, attr)
            if value is None:
                ret += '\n\n' + header.format(0)
            else:
                ret += '\n\n' + header.format(value.shape[0])
                ret += '\n' + serialize_array(value, row_len)
        return ret

    @classmethod
    def from_multi_mol(cls, multi_mol: MultiMolecule) -> PSF:
        """Construct :class:`PSF` instance from a :class:`.MultiMolecule` instance.

        Parameters
        ----------
        multi_mol : |FOX.MultiMolecule|_
            A :class:`.MultiMolecule` instance.

        Returns
        -------
        |FOX.PSF|_
            A new :class:`.PSF` instance constructed from **multi_mol**.

        """
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
        keys = sorted(set(df.loc[df['residue ID'] == 1, 'atom type']))
        values = range(1, 1 + len(keys))
        segment_dict = dict(zip(keys, values))
        value_max = 'MOL{:d}'.format(values.stop)

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
        """Update all ligand atom types and atomic charges in :attr:`PSF.atoms`.

        Performs an inplace update of the ``"charge"`` and ``"atom type"`` columns
        in :attr:`.PSF.atoms`.

        Parameters
        ----------
        filename : str
            The path+filename of a .str file containing ligand charges and atom types.

        """
        at_type, charge = read_str_file(filename)
        id_range = range(2, 1 + max(self.atoms['residue ID']))
        for i in id_range:
            j = self.atoms['residue ID'] == i
            self.atoms.loc[j, 'atom type'] = at_type
            self.atoms.loc[j, 'charge'] = charge

    def update_atom_charge(self, atom_type: str,
                           charge: float) -> None:
        """Change the charge of a specific atom type to **charge**.

        Performs an inplace update of the ``"charge"`` column in :attr:`PSF.atoms`.

        Parameters
        ----------
        atom_type : str
            An atom type in the ``"atom type"`` column in :attr:`.PSF.atoms`.

        charge : float
            A new atomic charge to-be associated with **atom_type**.

        """
        self.atoms.loc[self.atoms['atom type'] == atom_type, 'charge'] = charge
