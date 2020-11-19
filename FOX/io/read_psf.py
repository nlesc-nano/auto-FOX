"""A class for reading protein structure (.psf) files.

Index
-----
.. currentmodule:: FOX
.. autosummary::
    PSFContainer

API
---
.. autoclass:: PSFContainer
    :noindex:
    :members:

"""

import sys
import reprlib
import inspect
from os import PathLike
from typing import (Dict, Optional, Any, Set, Iterator, Iterable, TypeVar, Tuple, cast,
                    List, Mapping, Union, Collection, Generic, IO, Callable)
from itertools import chain
from types import MappingProxyType

import numpy as np
import pandas as pd

from scm.plams import Molecule, Atom, Bond
from assertionlib.dataclass import AbstractDataClass
from nanoutils import group_by_values, raise_if, AbstractFileContainer, set_docstring, TypedDict

from ..utils import read_str_file, read_rtf_file
from ..functions.molecule_utils import get_bonds, get_angles, get_dihedrals, get_impropers

try:
    from scm.plams import writepdb
    RDKIT_EX: Optional[ImportError] = None
except ImportError as ex:
    RDKIT_EX = ex

__all__ = ['PSFContainer']

T = TypeVar('T')


class DummyGetter(Generic[T]):
    def __init__(self, return_value: T) -> None:
        self.return_value = return_value

    def __getitem__(self, key: Any) -> T:
        return self.return_value

    def get(self, key: Any, default: Optional[Any] = None) -> T:
        return self.return_value


class _ShapeDictBase(TypedDict):
    shape: int


class _ShapeDict(_ShapeDictBase, total=False):
    row_len: int
    header: str


class PSFContainer(AbstractDataClass, AbstractFileContainer):
    r"""A container for managing protein structure files.

    The :class:`PSFContainer` class has access to three general sets of methods.

    Methods for reading & constructing .psf files:

        * :meth:`PSFContainer.read`
        * :meth:`PSFContainer.write`

    Methods for updating atom types:

        * :meth:`PSFContainer.update_atom_charge`
        * :meth:`PSFContainer.update_atom_type`

    Methods for extracting bond, angle and dihedral-pairs from |plams.Molecule| instances:

        * :meth:`PSFContainer.generate_bonds`
        * :meth:`PSFContainer.generate_angles`
        * :meth:`PSFContainer.generate_dihedrals`
        * :meth:`PSFContainer.generate_impropers`
        * :meth:`PSFContainer.generate_atoms`

    Attributes
    ----------
    filename : :math:`1` :class:`numpy.ndarray` [:class:`str`]
        A 1D array with a single string as filename.

    title : :math:`n` :class:`numpy.ndarray` [:class:`str`]
        A 1D array of strings holding the title block.

    atoms : :math:`n*8` :class:`pandas.DataFrame`
        A Pandas DataFrame holding the atoms block.
        The DataFrame should possess the following collumn keys:

        * ``"segment name"``
        * ``"residue ID"``
        * ``"residue name"``
        * ``"atom name"``
        * ``"atom type"``
        * ``"charge"``
        * ``"mass"``
        * ``"0"``

    bonds : :math:`n*2` :class:`numpy.ndarray` [:class:`int`]
        A 2D array holding the indices of all atom-pairs defining bonds.
        Indices are expected to be 1-based.

    angles : :math:`n*3` :class:`numpy.ndarray` [:class:`int`]
        A 2D array holding the indices of all atom-triplets defining angles.
        Indices are expected to be 1-based.

    dihedrals : :math:`n*4` :class:`numpy.ndarray` [:class:`int`]
        A 2D array holding the indices of all atom-quartets defining proper dihedral angles.
        Indices are expected to be 1-based.

    impropers : :math:`n*4` :class:`numpy.ndarray` [:class:`int`]
        A 2D array holding the indices of all atom-quartets defining improper dihedral angles.
        Indices are expected to be 1-based.

    donors : :math:`n*1` :class:`numpy.ndarray` [:class:`int`]
        A 2D array holding the atomic indices of all hydrogen-bond donors.
        Indices are expected to be 1-based.

    acceptors : :math:`n*1` :class:`numpy.ndarray` [:class:`int`]
        A 2D array holding the atomic indices of all hydrogen-bond acceptors.
        Indices are expected to be 1-based.

    no_nonbonded : :math:`n*2` :class:`numpy.ndarray` [:class:`int`]
        A 2D array holding the indices of all atom-pairs whose nonbonded
        interactions should be ignored.
        Indices are expected to be 1-based.

    np_printoptions : :class:`Mapping<collections.abc.Mapping>` [:class:`str`, :class:`object`]
        A mapping with Numpy print options.
        See `np.set_printoptions <https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html>`_.

    pd_printoptions : :class:`Mapping<collections.abc.Mapping>` [:class:`str`, :class:`object`]
        A mapping with Pandas print options.
        See `Options and settings <https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html>`_.

    """  # noqa

    #: A :class:`from` with the names of private instance attributes.
    #: These attributes will be excluded whenever calling :meth:`PSF.as_dict`.
    _PRIVATE_ATTR: Set[str] = frozenset({'_pd_printoptions', '_np_printoptions'})  # type: ignore[assignment]  # noqa: E501

    #: A dictionary containg array shapes among other things
    _SHAPE_DICT: Mapping[str, _ShapeDict] = MappingProxyType({
        'filename': {'shape': 1},
        'title': {'shape': 1},
        'atoms': {'shape': 8},
        'bonds': {'shape': 2, 'row_len': 4, 'header': '{:>10d} !NBOND: bonds'},
        'angles': {'shape': 3, 'row_len': 3, 'header': '{:>10d} !NTHETA: angles'},
        'dihedrals': {'shape': 4, 'row_len': 2, 'header': '{:>10d} !NPHI: dihedrals'},
        'impropers': {'shape': 4, 'row_len': 2, 'header': '{:>10d} !NIMPHI: impropers'},
        'donors': {'shape': 1, 'row_len': 8, 'header': '{:>10d} !NDON: donors'},
        'acceptors': {'shape': 1, 'row_len': 8, 'header': '{:>10d} !NACC: acceptors'},
        'no_nonbonded': {'shape': 2, 'row_len': 4, 'header': '{:>10d} !NNB'}
    })

    #: A dictionary mapping .psf headers to :class:`PSFContainer` attribute names
    _HEADER_DICT: Mapping[str, str] = MappingProxyType({
        '!NTITLE': 'title',
        '!NATOM': 'atoms',
        '!NBOND': 'bonds',
        '!NTHETA': 'angles',
        '!NPHI': 'dihedrals',
        '!NIMPHI': 'impropers',
        '!NDON': 'donors',
        '!NACC': 'acceptors',
        '!NNB': 'no_nonbonded'
    })

    def __init__(self, filename=None, title=None, atoms=None, bonds=None,
                 angles=None, dihedrals=None, impropers=None, donors=None,
                 acceptors=None, no_nonbonded=None) -> None:
        """Initialize a :class:`PSFContainer` instance.

        Parameters
        ----------
        filename : :math:`1` :class:`numpy.ndarray` [:class:`str`]
            Optional: A 1D array-like object containing a single filename.
            See also :attr:`PSFContainer.filename`.

        title : :math:`n` :class:`numpy.ndarray` [:class:`str`]
            Optional: A 1D array of strings holding the title block.
            See also :attr:`PSFContainer.title`.

        atoms : :math:`n*8` :class:`pandas.DataFrame`
            Optional: A Pandas DataFrame holding the atoms block.
            See also :attr:`PSFContainer.atoms`.

        bonds : :math:`n*2` :class:`numpy.ndarray` [:class:`int`]
            Optional: A 2D array-like object holding the indices of all atom-pairs defining bonds.
            See also :attr:`PSFContainer.bonds`.

        angles : :math:`n*3` :class:`numpy.ndarray` [:class:`int`]
            Optional: A 2D array-like object holding the indices of all atom-triplets defining angles.
            See also :attr:`PSFContainer.angles`.

        dihedrals : :math:`n*4` :class:`numpy.ndarray` [:class:`int`]
            Optional: A 2D array-like object holding the indices of
            all atom-quartets defining proper dihedral angles.
            See also :attr:`PSFContainer.dihedrals`.

        impropers : :math:`n*4` :class:`numpy.ndarray` [:class:`int`]
            Optional: A 2D array-like object holding the indices of
            all atom-quartets defining improper dihedral angles.
            See also :attr:`PSFContainer.impropers`.

        donors : :math:`n*1` :class:`numpy.ndarray` [:class:`int`]
            Optional: A 2D array-like object holding the atomic indices of all hydrogen-bond donors.
            See also :attr:`PSFContainer.donors`.

        acceptors : :math:`n*1` :class:`numpy.ndarray` [:class:`int`]
            Optional: A 2D array-like object holding the atomic indices of all hydrogen-bond acceptors.
            See also :attr:`PSFContainer.acceptors`.

        no_nonbonded : :math:`n*2` :class:`numpy.ndarray` [:class:`int`]
            Optional: A 2D array-like object holding the indices of all atom-pairs whose nonbonded
            interactions should be ignored.
            See also :attr:`PSFContainer.no_nonbonded`.

        """  # noqa: E501
        super().__init__()

        self.filename = filename
        self.title = title
        self.atoms = atoms
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.impropers = impropers
        self.donors = donors
        self.acceptors = acceptors
        self.no_nonbonded = no_nonbonded

        # Print options for NumPy ndarrays and Pandas DataFrames
        self.np_printoptions: Dict[str, Any] = {'threshold': 20, 'edgeitems': 5}
        self.pd_printoptions: Dict[str, Any] = cast(
            Iterator[Tuple[str, Any]], {'display.max_rows': 10}
        )

    @property
    def np_printoptions(self) -> Dict[str, Any]:
        return self._np_printoptions

    @np_printoptions.setter
    def np_printoptions(self, value: Dict[str, Any]) -> None:
        self._np_printoptions = self._is_dict(value)

    @property
    def pd_printoptions(self) -> Iterator[Tuple[str, Any]]:
        return chain.from_iterable(self._pd_printoptions.items())

    @pd_printoptions.setter
    def pd_printoptions(self, value: Dict[str, Any]) -> None:
        self._pd_printoptions = self._is_dict(value)

    @staticmethod
    def _is_dict(value: Any) -> dict:
        """Check if **value** is a :class:`dict` instance; raise a :exc:`TypeError` if not."""
        if not isinstance(value, dict):
            caller_name: str = inspect.stack()[1][3]
            raise TypeError(f"The {caller_name!r} parameter expects an instance of 'dict'; "
                            f"observed type: {value.__class__.__name__!r}")
        return value

    @AbstractDataClass.inherit_annotations()
    def __repr__(self):
        with np.printoptions(**self.np_printoptions), pd.option_context(*self.pd_printoptions):
            return super().__repr__()

    @AbstractDataClass.inherit_annotations()
    def _str_iterator(self): return ((k.strip('_'), v) for k, v in super()._str_iterator())

    @AbstractDataClass.inherit_annotations()
    def __eq__(self, value):
        if type(self) is not type(value):
            return False

        try:
            for k, v in vars(self).items():
                if k in self._PRIVATE_ATTR:
                    continue
                v1 = np.asarray(v)
                v2 = np.asarray(getattr(value, k))
                assert (v1 == v2).all()
        except (AttributeError, AssertionError):
            return False
        else:
            return True

    @AbstractDataClass.inherit_annotations()
    def as_dict(self, return_private=False):
        ret = super().as_dict(return_private)
        return {k.strip('_'): v for k, v in ret.items()}

    # Ensure that a deepcopy is returned unless explictly specified

    @AbstractDataClass.inherit_annotations()
    def copy(self, deep=True): return super().copy(deep)

    @AbstractDataClass.inherit_annotations()
    def __copy__(self): return self.copy(deep=True)

    """###################################### Properties ########################################"""

    @property
    def filename(self) -> str:
        """Get :attr:`PSFContainer.filename` as string or assign an array-like object as a 1D array."""  # noqa
        filename = self._filename  # type: ignore[attr-defined]
        return str(filename[0]) if len(filename) else str(filename)

    @filename.setter
    def filename(self, value: Iterable): self._set_nd_array('_filename', value, 1, str)

    @property
    def title(self) -> np.ndarray:
        """Get :attr:`PSFContainer.title` or assign an array-like object as a 1D array."""
        return self._title  # type: ignore[attr-defined]

    @title.setter
    def title(self, value: Iterable):
        if value is not None:
            self._set_nd_array('_title', value, 1, str)
        else:
            self._title: np.ndarray = np.array(['PSF file generated with Auto-FOX',
                                                'https://github.com/nlesc-nano/Auto-FOX'])

    @property
    def atoms(self) -> pd.DataFrame:
        """Get :attr:`PSFContainer.atoms` or assign an a DataFrame."""
        return self._atoms

    @atoms.setter
    def atoms(self, value: Iterable):
        self._atoms = value if value is not None else pd.DataFrame()

    @property
    def bonds(self) -> np.ndarray:
        """Get :attr:`PSFContainer.bonds` or assign an array-like object as a 2D array."""
        return self._bonds  # type: ignore[attr-defined]

    @bonds.setter
    def bonds(self, value: Iterable):
        self._set_nd_array('_bonds', value, 2, int)

    @property
    def angles(self) -> np.ndarray:
        """Get :attr:`PSFContainer.angles` or assign an array-like object as a 2D array."""
        return self._angles  # type: ignore[attr-defined]

    @angles.setter
    def angles(self, value: Iterable):
        self._set_nd_array('_angles', value, 2, int)

    @property
    def dihedrals(self) -> np.ndarray:
        """Get :attr:`PSFContainer.dihedrals` or assign an array-like object as a 2D array."""
        return self._dihedrals  # type: ignore[attr-defined]

    @dihedrals.setter
    def dihedrals(self, value: Iterable):
        self._set_nd_array('_dihedrals', value, 2, int)

    @property
    def impropers(self) -> np.ndarray:
        """Get :attr:`PSFPSFContainerimpropers` or assign an array-like object as a 2D array."""
        return self._impropers  # type: ignore[attr-defined]

    @impropers.setter
    def impropers(self, value: Iterable):
        self._set_nd_array('_impropers', value, 2, int)

    @property
    def donors(self) -> np.ndarray:
        """Get :attr:`PSFContainer.donors` or assign an array-like object as a 2D array."""
        return self._donors  # type: ignore[attr-defined]

    @donors.setter
    def donors(self, value: Iterable):
        self._set_nd_array('_donors', value, 2, int)

    @property
    def acceptors(self) -> np.ndarray:
        """Get :attr:`PSFContainer.acceptors` or assign an array-like object as a 2D array."""
        return self._acceptors  # type: ignore[attr-defined]

    @acceptors.setter
    def acceptors(self, value: Iterable):
        self._set_nd_array('_acceptors', value, 2, int)

    @property
    def no_nonbonded(self) -> np.ndarray:
        """Get :attr:`PSFContainer.no_nonbonded` or assign an array-like object as a 2D array."""
        return self._no_nonbonded  # type: ignore[attr-defined]

    @no_nonbonded.setter
    def no_nonbonded(self, value: Iterable):
        self._set_nd_array('_no_nonbonded', value, 2, int)

    def _set_nd_array(self, name: str, value: Optional[np.ndarray],
                      ndmin: int, dtype: type) -> None:
        """Assign an array-like object (**value**) to the **name** attribute as ndarray.

        Performs an inplace update of this instance.

        .. _`array-like`: https://docs.scipy.org/doc/numpy/glossary.html#term-array-like

        Parameters
        ----------
        name : :class:`str`
            The name of the to-be set attribute.

        value : `array-like`_
            The array-like object to-be assigned to **name**.
            The supplied object is converted into into an array beforehand.

        ndmin : :class:`int`
            The minimum number of dimensions of the to-be assigned array.

        dtype : :class:`type` or :class:`numpy.dtype`
            The desired datatype of the to-be assigned array.

        Exceptions
        ----------
        ValueError:
            Raised if value array construction was unsuccessful.

        """
        _value = [] if value is None else value

        try:
            array = np.array(_value, dtype=dtype, ndmin=ndmin, copy=False)
        except TypeError:  # **value** is an iterator
            array = np.fromiter(_value, dtype=dtype)

        try:
            setattr(self, name, array)
        except ValueError as ex:
            _name = name.strip('_')
            ex.args = (f"The parameter '{_name}' expects a {ndmin}d array-like object consisting "
                       f"of '{dtype}'; observed type: '{value.__class__.__name__}'",)
            raise ex

    """################################## dataframe shortcuts ###################################"""

    @property
    def segment_name(self) -> pd.Series:
        """Get or set the ``"segment name"`` column in :attr:`PSFContainer.atoms`."""
        return self.atoms['segment name']

    @segment_name.setter
    def segment_name(self, value) -> None: self.atoms['segment name'] = value

    @property
    def residue_id(self) -> pd.Series:
        """Get or set the ``"residue ID"`` column in :attr:`PSFContainer.atoms`."""
        return self.atoms['residue ID']

    @residue_id.setter
    def residue_id(self, value) -> None: self.atoms['residue ID'] = value

    @property
    def residue_name(self) -> pd.Series:
        """Get or set the ``"residue name"`` column in :attr:`PSFContainer.atoms`."""
        return self.atoms['residue name']

    @residue_name.setter
    def residue_name(self, value) -> None: self.atoms['residue name'] = value

    @property
    def atom_name(self) -> pd.Series:
        """Get or set the ``"atom name"`` column in :attr:`PSFContainer.atoms`."""
        return self.atoms['atom name']

    @atom_name.setter
    def atom_name(self, value) -> None: self.atoms['atom name'] = value

    @property
    def atom_type(self) -> pd.Series:
        """Get or set the ``"atom type"`` column in :attr:`PSFContainer.atoms`."""
        return self.atoms['atom type']

    @atom_type.setter
    def atom_type(self, value) -> None: self.atoms['atom type'] = value

    @property
    def charge(self) -> pd.Series:
        """Get or set the ``"charge"`` column in :attr:`PSFContainer.atoms`."""
        return self.atoms['charge']

    @charge.setter
    def charge(self, value) -> None: self.atoms['charge'] = value

    @property
    def mass(self) -> pd.Series:
        """Get or set the ``"mass"`` column in :attr:`PSFContainer.atoms`."""
        return self.atoms['mass']

    @mass.setter
    def mass(self, value) -> None: self.atoms['mass'] = value

    """########################### methods for reading .psf files. ##############################"""

    @classmethod
    @set_docstring(AbstractFileContainer.read.__doc__)
    def _read(cls, file_obj, decoder):
        ret = {}

        next(file_obj)  # Skip the first line
        for _i in file_obj:
            i = decoder(_i)

            # Search for psf blocks
            if i == '\n':
                continue

            # Read the psf block header
            try:
                key = cls._HEADER_DICT[i.split()[1].rstrip(':')]
            except KeyError as ex:
                err = f'Failed to parse file; invalid header: {reprlib.repr(i)}'
                raise OSError(err).with_traceback(ex.__traceback__)
            ret[key] = value = []

            # Read the actual psf blocks
            try:
                j = next(file_obj)
            except StopIteration:
                break

            while j != '\n':
                value.append(j.split())
                try:
                    j = next(file_obj)
                except StopIteration:
                    break

        return cls._post_process_psf(ret)

    @classmethod
    def _post_process_psf(cls, psf_dict: dict) -> Dict[str, np.ndarray]:
        """Post-process the output of :meth:`PSF.read`, casting the values into appropiat objects.

        * The title block is converted into a 1D array of strings.
        * The atoms block is converted into a Pandas DataFrame.
        * All other blocks are converted into 2D arrays of integers.

        Parameters
        ----------
        psf_dict : :class:`dict` [:class:`str`, :class:`numpy.ndarray`]
            A dictionary holding the content of a .psf file (see :func:`PSFContainer.read_psf`).

        Returns
        -------
        :class:`dict` [:class:`str`, :class:`numpy.ndarray`]:
            The .psf output, **psf_dict**, with properly formatted values.

        """
        for key, value in psf_dict.items():  # Post-process the output
            # Cast the atoms block into a dataframe
            if key == 'atoms':
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

            # Cast into a flattened array of indices
            else:
                ar = np.fromiter(chain.from_iterable(value), dtype=int)
                ar.shape = len(ar) // cls._SHAPE_DICT[key]['shape'], cls._SHAPE_DICT[key]['shape']
                psf_dict[key] = ar

        return psf_dict

    """########################### methods for writing .psf files. ##############################"""

    @set_docstring(AbstractFileContainer._write.__doc__)
    def _write(self, file_obj, encoder):
        self._write_top(file_obj, encoder)
        self._write_bottom(file_obj, encoder)

    def _write_top(self, file_obj, encoder) -> None:
        """Write the top-most section of the to-be create .psf file.

        The following blocks are seralized:

            * :attr:`PSF.title`
            * :attr:`PSF.atoms`

        Returns
        -------
        :class:`str`
            A string constructed from the above-mentioned psf blocks.

        See Also
        --------
        :meth:`PSFContainer.write`
            The main method for writing .psf files.

        """
        write = lambda n: file_obj.write(encoder(n))  # noqa

        # Prepare the !NTITLE block
        write('PSF EXT\n\n{:>10d} !NTITLE\n'.format(self.title.shape[0]))
        for i in self.title:
            write(f'   REMARKS {i}\n')

        # Prepare the !NATOM block
        write('\n\n{:>10d} !NATOM\n'.format(self.atoms.shape[0]))
        string = '{:>10d} {:8.8} {:<8d} {:8.8} {:8.8} {:6.6} {:>9f} {:>15f} {:>8d}\n'
        for i, j in self.atoms.iterrows():
            args = [i] + j.values.tolist()
            write(string.format(*args))

    def _write_bottom(self, file_obj, encoder) -> None:
        """Write the bottom-most section of the to-be create .psf file.

        The following blocks are seralized:

            * :attr:`PSF.bonds`
            * :attr:`PSF.angles`
            * :attr:`PSF.dihedrals`
            * :attr:`PSF.impropers`
            * :attr:`PSF.donors`
            * :attr:`PSF.acceptors`
            * :attr:`PSF.no_nonbonded`

        See Also
        --------
        :meth:`PSFContainer.write`
            The main method for writing .psf files.

        """
        write = lambda n: file_obj.write(encoder(n))  # noqa: E731
        sections = ('bonds', 'angles', 'dihedrals', 'impropers',
                    'donors', 'acceptors', 'no_nonbonded')

        for attr in sections:
            header = self._SHAPE_DICT[attr]['header']
            row_len = self._SHAPE_DICT[attr]['row_len']

            value = getattr(self, attr)
            item_count = len(value) if value.shape[-1] != 0 else 0
            write('\n\n' + header.format(item_count) +
                  '\n' + self._serialize_array(value, row_len))

    @staticmethod
    def _serialize_array(array: np.ndarray, items_per_row: int = 4) -> str:
        """Serialize an array into a single string; used for creating .psf files.

        Newlines are placed for every **items_per_row** rows in **array**.

        Parameters
        ----------
        array : :class:`numpy.ndarray`
            A 2D array.

        items_per_row : :class:`int`
            The number of values per row before switching to a new line.

        Returns
        -------
        :class:`str`:
            A serialized array.

        See Also
        --------
        :meth:`PSFContainer.write`
            The main method for writing .psf files.

        """
        if len(array) == 0:
            return ''

        ret = ''
        k = 0
        for i in array:
            for j in i:
                ret += '{:>10d}'.format(j)
            k += 1
            if k == items_per_row:
                k = 0
                ret += '\n'

        return ret

    """################### methods for altering atomic/molecular information. ###################"""

    def update_atom_charge(self, atom_type: str, charge: float) -> None:
        """Change the charge of **atom_type** to **charge**.

        Parameters
        ----------
        atom_type : :class:`str`
            An atom type in :attr:`PSFContainer.atoms` ``["atom type"]``.

        charge : :class:`float`
            The new atomic charge to-be assigned to **atom_type**.
            See :attr:`PSFContainer.atoms` ``["charge"]``.

        Raises
        ------
        ValueError
            Raised if **charge** cannot be converted into a :class:`float`.

        """
        condition = self.atom_type == atom_type
        self.atoms.loc[condition, 'charge'] = float(charge)

    def update_atom_type(self, atom_type_old: str, atom_type_new: str) -> None:
        """Change the atom type of a **atom_type_old** to **atom_type_new**.

        Parameters
        ----------
        atom_type_old : :class:`str`
            An atom type in :attr:`PSFContainer.atoms` ``["atom type"]``.

        atom_type_new : :class:`str`
            The new atom type to-be assigned to **atom_type**.
            See :attr:`PSFContainer.atoms` ``["atom type"]``.

        """
        condition = self.atom_type == atom_type_old
        self.atoms.loc[condition, 'atom type'] = atom_type_new

    def generate_bonds(self, mol: Molecule) -> None:
        """Update :attr:`PSFContainer.bonds` with the indices of all bond-forming atoms from **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|
            A PLAMS Molecule.

        """  # noqa
        self.bonds = get_bonds(mol)

    def generate_angles(self, mol: Molecule) -> None:
        """Update :attr:`PSFContainer.angles` with the indices of all angle-defining atoms from **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|
            A PLAMS Molecule.

        """  # noqa
        self.angles = get_angles(mol)

    def generate_dihedrals(self, mol: Molecule) -> None:
        """Update :attr:`PSFContainer.dihedrals` with the indices of all proper dihedral angle-defining atoms from **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|
            A PLAMS Molecule.

        """  # noqa
        self.dihedrals = get_dihedrals(mol)

    def generate_impropers(self, mol: Molecule) -> None:
        """Update :attr:`PSFContainer.impropers` with the indices of all improper dihedral angle-defining atoms from **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|
            A PLAMS Molecule.

        """  # noqa
        self.impropers = get_impropers(mol)

    def generate_atoms(self, mol: Molecule,
                       id_map: Optional[Mapping[int, Any]] = None) -> None:
        """Update :attr:`PSFContainer.atoms` with the all properties from **mol**.

        DataFrame keys in :attr:`PSFContainer.atoms` are set based on the following values in **mol**:

        ================== ========================================================= =================================================
        DataFrame column   Value                                                     Backup value(s)
        ================== ========================================================= =================================================
        ``"segment name"`` ``"MOL{:d}"``; See ``"atom type"`` and ``"residue name"``
        ``"residue ID"``   |Atom.properties| ``["pdb_info"]["ResidueNumber"]``       ``1``
        ``"residue name"`` |Atom.properties| ``["pdb_info"]["ResidueName"]``         ``"COR"``
        ``"atom name"``    |Atom.symbol|
        ``"atom type"``    |Atom.properties| ``["symbol"]``                          |Atom.symbol|
        ``"charge"``       |Atom.properties| ``["charge_float"]``                    |Atom.properties| ``["charge"]`` & ``0.0``
        ``"mass"``         |Atom.mass|
        ``"0"``            ``0``
        ================== ========================================================= =================================================

        If a value is not available in a particular |Atom.properties| instance then
        a backup value will be set.

        Parameters
        ----------
        mol : |plams.Molecule|
            A PLAMS Molecule.

        id_map : :class:`Mapping<collection.abc.Mapping>` [:class:`int`, :data:`Hashable<typing.Hashable>`], optional
            A mapping of ligand residue ID's to a custom (Hashable) descriptor.
            Can be used for generating residue names for quantum dots with
            multiple different ligands.

        """  # noqa
        def get_res_id(at: Atom) -> int:
            return at.properties.get('pdb_info', {}).get('ResidueNumber', 1)

        def get_res_name(at: Atom) -> str:
            return at.properties.get('pdb_info', {}).get('ResidueName', 'COR')

        def get_at_type(at: Atom) -> str:
            return at.properties.get('symbol', at.symbol)

        def get_charge(at: Atom) -> float:
            properties = at.properties
            if 'charge_float' in properties:
                return float(properties.charge_float)
            elif 'charge' in properties:
                return float(properties.charge)
            return 0.0

        index = pd.RangeIndex(1, 1 + len(mol))
        self.atoms = df = pd.DataFrame(index=index, dtype=str)

        df['segment name'] = None
        df['residue ID'] = [get_res_id(at) for at in mol]
        df['residue name'] = [get_res_name(at) for at in mol]
        df['atom name'] = [at.symbol for at in mol]
        df['atom type'] = [get_at_type(at) for at in mol]
        df['charge'] = [get_charge(at) for at in mol]
        df['mass'] = [at.mass for at in mol]
        df['0'] = 0

        df['segment name'] = self._construct_segment_name(id_map)

    def _construct_segment_name(self, id_map: Optional[Mapping[int, Any]] = None) -> List[str]:
        """Generate a list for the :attr:`PSF.atoms` ``["segment name"]`` column."""
        ret: List[str] = []
        ret_append = ret.append

        type_dict: Dict[str, str] = {}
        id_map_get = cast(
            Callable[[int, str], str],
            id_map.get if id_map is not None else DummyGetter('LIG').get
        )

        iterator = zip(self.atom_type, self.residue_name, self.residue_id)
        for at_type, res_name, res_id in iterator:
            if res_name == 'LIG':
                at_type = id_map_get(res_id, 'LIG')

            try:
                value = type_dict[at_type]
            except KeyError:
                type_dict[at_type] = value = 'MOL{:d}'.format(1 + len(type_dict))

            ret_append(value)

        return ret

    def to_atom_dict(self) -> Dict[str, List[int]]:
        """Create a dictionary of atom types and lists with their respective indices.

        Returns
        -------
        :class:`dict` [:class:`str`, :class:`list` [:class:`int`]]
            A dictionary with atom types as keys and lists of matching atomic indices as values.
            The indices are 0-based.

        """
        return group_by_values(enumerate(self.atom_type))

    @raise_if(RDKIT_EX)
    def write_pdb(self, mol: Molecule,
                  pdb_file: Union[str, PathLike, IO[str]] = sys.stdout,
                  copy_mol: bool = True) -> None:
        """Construct a .pdb file from this instance and **mol**.

        Parameters
        ----------
        mol : |plams.Molecule|
            A PLAMS Molecule.

        copy_mol : :class:`bool`
            If ``True``, create a copy of **mol** instead of modifying it inplace.

        pdb_file : :class:`str` ot :class:`TextIOBase<io.TextIOBase>`
            A filename or a file-like object.

        """
        if hasattr(pdb_file, '__fspath__'):
            pdb_file = str(pdb_file)
        if copy_mol:
            mol = mol.copy()

        # Update residue information and the like
        for at, (_, series) in zip(mol, self.atoms.iterrows()):
            at.properties.pdb_info += {
                'ResidueNumber': series['residue ID'],
                'ResidueName': series['residue name'],
                'Name': series['atom type'],
                'IsHeteroAtom': at.symbol not in {'C', 'H'},
                'ChainId': 'A'
            }

        # Update bonds
        mol.delete_all_bonds()
        for i, j in self.bonds:
            mol.add_bond(Bond(mol[i], mol[j], mol=mol))

        writepdb(mol, pdb_file)


def overlay_str_file(psf: PSFContainer, filename: Union[str, bytes, PathLike],
                     id_range: Optional[Iterable[int]] = None) -> None:
    """Update all ligand atom types and atomic charges in :attr:`PSF.atoms`.

    Performs an inplace update of the ``"charge"`` and ``"atom type"`` columns
    in :attr:`PSFContainer.atoms`.

    Parameters
    ----------
    psf : |FOX.PSFContainer|
        A :class:`PSFContainer` instance.

    filename : str
        The path+filename of a .str file containing ligand charges and atom types.

    id_range : :class:`Iterable<collections.abc.Iterable>` [:class:`int`], optional
        An iterable with the residue IDs of to-be updated residues.
        If ``None``, update the atoms in all residues.

    """
    try:
        atoms, charge = read_str_file(filename)  # type: ignore
    except TypeError as ex:
        raise RuntimeError(f"Failed to parse {filename!r}") from ex
    _overlay(psf, atoms, charge, id_range)


def overlay_rtf_file(psf: PSFContainer, filename: Union[str, bytes, PathLike],
                     id_range: Optional[Iterable[int]] = None) -> None:
    """Update all ligand atom types and atomic charges in :attr:`PSF.atoms`.

    Performs an inplace update of the ``"charge"`` and ``"atom type"`` columns
    in :attr:`PSFContainer.atoms`.

    Parameters
    ----------
    psf : |FOX.PSFContainer|
        A :class:`PSFContainer` instance.

    filename : str
        The path+filename of a .rtf file containing ligand charges and atom types.

    id_range : :class:`Iterable<collections.abc.Iterable>` [:class:`int`], optional
        An iterable with the residue IDs of to-be updated residues.
        If ``None``, update the atoms in all residues.

    """
    try:
        atoms, charge = read_rtf_file(filename)  # type: ignore
    except TypeError as ex:
        raise RuntimeError(f"Failed to parse {filename!r}") from ex
    _overlay(psf, atoms, charge, id_range)


def _overlay(psf: PSFContainer, atom_list: Collection[str],
             charge_list: Collection[float],
             id_range: Optional[Iterable[int]] = None) -> None:
    id_range = range(2, 1 + max(psf.residue_id)) if id_range is None else id_range
    try:
        for i in id_range:
            j = psf.residue_id == i
            psf.atoms.loc[j, 'atom type'] = atom_list
            psf.atoms.loc[j, 'charge'] = charge_list

    except ValueError as ex:
        try:
            ligand_size = np.count_nonzero(j)
            at_len = len(atom_list)
            charge_len = len(charge_list)
        except Exception as ex2:  # Plan B in case something goes wrong here
            raise ex from ex2

        if ligand_size != at_len:
            raise ValueError(f"Residue {i} in the passed {psf.__class__.__name__} contains "
                             f"{ligand_size} atoms while the passed 'atom_list' "
                             f"contains {at_len}") from ex
        elif ligand_size != charge_len:
            raise ValueError(f"Residue {i} in the passed {psf.__class__.__name__} contains "
                             f"{ligand_size} atoms while the passed 'charge_list' "
                             f"contains {charge_len}") from ex
        raise ex
