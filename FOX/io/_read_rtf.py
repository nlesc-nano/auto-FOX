"""A class for reading and CHARMM .rtf topology files.

Index
-----
.. currentmodule:: FOX
.. autosummary::
    RTFContainer
    RTFContainer.collapse_charges
    RTFContainer.auto_to_explicit
    RTFContainer.from_file
    RTFContainer.concatenate

API
---
.. autoclass:: RTFContainer
    :noindex:
    :members: mass, atom, bond, impropers, angles, dihedrals, charmm_version, auto

.. automethod:: RTFContainer.collapse_charges
.. automethod:: RTFContainer.auto_to_explicit
.. automethod:: RTFContainer.from_file
.. automethod:: RTFContainer.concatenate

"""
from __future__ import annotations

import os
import types
import textwrap
import itertools
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from collections.abc import Mapping, Iterator, Iterable
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from scm.plams import Molecule, Atom

from . import FileIter
from ..functions.molecule_utils import get_angles, get_dihedrals

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import NDArray

__all__ = ["RTFContainer"]


class RTFContainer:
    """A class for managing CHARMM .rtf topology files.

    Examples
    --------
    .. code:: python

        >>> from FOX import RTFContainer

        >>> input_file = str(...)
        >>> rtf = RTFContainer.from_file(input_file)

    """

    __slots__ = (
        "__weakref__",
        "mass",
        "atom",
        "bond",
        "impr",
        "angles",
        "dihe",
        "charmm_version",
        "auto",
        "_pd_printoptions",
    )

    #: A dataframe holding all MASS-related info.
    mass: pd.DataFrame
    #: A dataframe holding all ATOM-related info.
    atom: pd.DataFrame
    #: A dataframe holding all BOND-related info.
    bond: pd.DataFrame
    #: A dataframe holding all IMPR-related info.
    impr: pd.DataFrame
    #: A dataframe holding all ANGLES-related info.
    angles: pd.DataFrame
    #: A dataframe holding all DIHE-related info.
    dihe: pd.DataFrame
    #: The CHARMM version used for generating the .rtf file
    charmm_version: tuple[int, ...]
    #: A set with all .rtf statements that should be auto-generated.
    auto: set[str]
    #: Print options as used by :meth:`~RTFContainer.__repr__`.
    _pd_printoptions: dict[str, Any]

    #: A mapping with strucutred dtypes for each dataframe column and index.
    DTYPES: ClassVar[types.MappingProxyType[str, np.dtype[np.void]]] = types.MappingProxyType({
        "MASS": np.dtype([
            ("index", "i8"),
            ("atom_type", "U5"),
            ("mass", "f8"),
            ("atom_name", "U2"),
        ]),
        "ATOM": np.dtype([
            ("molecule", "U5"),
            ("atom1", "i8"),
            ("atom_type", "U5"),
            ("charge", "f8"),
        ]),
        "BOND": np.dtype([
            ("molecule", "U5"),
            ("atom1", "i8"),
            ("atom2", "i8"),
        ]),
        "ANGLES": np.dtype([
            ("molecule", "U5"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("atom3", "i8"),
        ]),
        "DIHE": np.dtype([
            ("molecule", "U5"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("atom3", "i8"),
            ("atom4", "i8"),
        ]),
        "IMPR": np.dtype([
            ("molecule", "U5"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("atom3", "i8"),
            ("atom4", "i8"),
        ]),
    })

    @property
    def impropers(self) -> pd.DataFrame:
        """A dataframe holding all IMPR-related info."""
        return self.impr

    @impropers.setter
    def impropers(self, value: pd.DataFrame) -> None:
        self.impr = value

    @property
    def dihedrals(self) -> pd.DataFrame:
        """A dataframe holding all DIHE-related info."""
        return self.dihe

    @dihedrals.setter
    def dihedrals(self, value: pd.DataFrame) -> None:
        self.dihe = value

    @property
    def pd_printoptions(self) -> Iterator[Any]:
        """Return an iterator flattening :attr:`_pd_printoptions`."""
        return itertools.chain.from_iterable(self._pd_printoptions.items())

    @property
    def residues(self) -> pd.Index:
        """Get all unique residue names."""
        return self.atom.index[~self.atom.index.duplicated()]

    def __init__(
        self,
        mass: pd.DataFrame,
        atom: pd.DataFrame,
        bond: pd.DataFrame,
        impr: pd.DataFrame,
        angles: pd.DataFrame,
        dihe: pd.DataFrame,
        charmm_version: tuple[int, ...] = (0, 0),
        auto: None | set[str] = None,
    ) -> None:
        """Initialize the instance."""
        self.mass = mass
        self.atom = atom
        self.bond = bond
        self.impropers = impr
        self.angles = angles
        self.dihedrals = dihe
        self.charmm_version = charmm_version
        self.auto = auto if auto is not None else set()
        self._pd_printoptions = {"display.max_rows": 20}

    def __eq__(self, other: object) -> bool:
        """Implement :meth:`self == other <object.__eq__>`."""
        cls = type(self)
        if not isinstance(other, cls):
            return NotImplemented

        if self.auto != other.auto:
            return False

        df_keys = ["mass", "atom", "bond", "impropers", "angles", "dihedrals"]
        iterator = ((getattr(self, k), getattr(other, k)) for k in df_keys)
        return all(df1.equals(df2) for df1, df2 in iterator)

    def __reduce__(self) -> tuple[type[Self], tuple[Any, ...]]:
        """Helper function for :mod:`pickle`."""
        cls = type(self)
        return cls, (
            self.mass,
            self.atom,
            self.bond,
            self.impropers,
            self.angles,
            self.dihedrals,
            self.charmm_version,
            self.auto,
        )

    def __repr__(self) -> str:
        """Implement :func:`repr(self)<repr>`."""
        # Get all to-be printed attribute (names)
        cls = type(self)
        attr_names = ["mass", "atom", "bond", "impropers", "angles", "dihedrals"]

        # Determine the indentation width
        width = max(len(k) for k in attr_names)
        indent = width + 3

        # Gather string representations of all attributes
        ret = ""
        with pd.option_context(*self.pd_printoptions):
            items = ((k, getattr(self, k)) for k in attr_names)
            for k, _v in items:
                v = textwrap.indent(repr(_v), " " * indent)[indent:]
                ret += f"{k:{width}} = {v},\n"
            ret += f"{'auto':{width}} = {self.auto!r},\n"
            ret += f"{'charmm_version':{width}} = {self.charmm_version!r},\n"
        return f"{cls.__name__}(\n{textwrap.indent(ret[:-2], 4 * ' ')}\n)"

    def collapse_charges(self) -> dict[str, float]:
        """Return a dictionary mapping atom types to atomic charges.

        Returns
        -------
        dict[str, float]

        Raises
        ------
        ValueError:
            Raised if an atom type has multiple unique charges associated with it

        """
        dct: dict[str, set[float]] = defaultdict(set)
        for at, charge in zip(self.atom["atom_type"], self.atom["charge"].round(6)):
            dct[at].add(charge)

        illegal = {k: sorted(v) for k, v in dct.items() if len(v) > 1}
        if illegal:
            raise ValueError(
                f"Found {len(illegal)} atom types with two or more "
                f"distinct charges: {illegal!r}"
            )
        return {k: v.pop() for k, v in dct.items()}

    def auto_to_explicit(self) -> None:
        """Convert all statements in :attr:`~RTFContainer.auto` into explicit dataframe."""
        if not self.auto:
            return

        # Construct a dictionary mapping residue names to PLAMS molecules (with bonds)
        atom_dict: dict[str, str] = dict(zip(self.mass["atom_type"], self.mass["atom_name"]))
        mol_dict: dict[str, Molecule] = {}
        for res in self.residues:
            mol_dict[res] = mol = Molecule()
            for at_type in self.atom["atom_type"]:
                mol.add_atom(Atom(symbol=atom_dict[at_type]))
            for (i, j) in zip(self.bond["atom1"], self.bond["atom2"]):
                mol.add_bond(mol[i], mol[j])

        # Generate angles and/or proper dihedral angles based on the AUTO settings
        if "ANGLES" in self.auto:
            self.angles = self._auto_to_explicit("ANGLES", mol_dict)
            self.auto.remove("ANGLES")
        if "DIHE" in self.auto:
            self.dihedrals = self._auto_to_explicit("DIHE", mol_dict)
            self.auto.remove("DIHE")
        if self.auto:
            warnings.warn(f"Unsupported auto statements: {sorted(self.auto)!r}", stacklevel=2)

    def _auto_to_explicit(
        self,
        key: Literal["ANGLES", "DIHE"],
        mol_dict: Mapping[str, Molecule],
    ) -> pd.DataFrame:
        if key == "ANGLES":
            func = get_angles
        elif key == "DIHE":
            func = get_dihedrals
        else:
            raise ValueError(key)
        dtype = self.DTYPES[key]
        assert dtype.names is not None

        # Computer the angles/dihedrals for all molecules
        array_dict = {}
        for res, mol in mol_dict.items():
            array_dict[res] = func(mol)

        # Concatenate the residue-specific angles/dihedrals into a single structured array
        i = j = 0
        total_array = np.empty(sum(len(i) for i in array_dict.values()), dtype=dtype)
        for res, array in array_dict.items():
            j += len(array)
            total_array["molecule"][i:j] = res
            for k, field_name in enumerate(dtype.names[1:]):
                total_array[field_name][i:j] = array[..., k]
            i += len(array)

        # Convert the strucutred array into a dataframe
        df = pd.DataFrame(total_array)
        df.set_index("molecule", inplace=True, drop=True)
        return df

    def _to_hdf5_dict(self) -> dict[str, NDArray[np.void]]:
        dct: dict[str, NDArray[np.void]] = {}
        for name, _dtype in self.DTYPES.items():
            assert _dtype.fields is not None

            # Construct a h5py-compatible structured dtype
            dtype_list = []
            for sub_field, (sub_dtype, *_) in _dtype.fields.items():
                if sub_dtype.kind == "U":
                    sub_dtype = h5py.string_dtype("utf-8", sub_dtype.itemsize // 4)
                dtype_list.append((sub_field, sub_dtype))
            dtype = np.dtype(dtype_list)

            df: pd.DataFrame = getattr(self, name.lower()).reset_index(inplace=False, drop=False)
            dct[name] = df.to_records(index=False).astype(dtype)
        return dct

    @classmethod
    def _get_err_msg(cls, statement: str, lst: list[tuple[Any, ...]]) -> None | str:
        """Construct an error message for when :meth:`~RTFContainer.from_file` fails to \
        construct an array.

        Parameters
        ----------
        statement : str
            The name of the match statement
        lst : list[tuple[Any, ...]]
            A list of tuples with structured data.
            The first field is guaranteed to be the residue name (a string)

        Returns
        -------
        str | None
            A newly constructed error message or :data:`None` if one could not be constructed

        """
        dtype = cls.DTYPES[statement]
        i = 0
        residue_old = ""
        for tup in lst:
            residue: str = tup[0]
            if residue != residue_old:
                i = 1
            else:
                i += 1
            residue_old = residue
            try:
                np.array(tup, dtype=dtype)
            except Exception:
                return f"failed to parse {statement!r} statement {i} in residue {residue!r}"
        return None

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Construct a new :class:`RTFContainer` from the passed file path.

        Parameters
        ----------
        path : path-like object
            The path to the .rtf file

        Returns
        -------
        FOX.RTFContaier
            A newly constructed .rtf container

        """
        dct: dict[str, list[tuple[Any, ...]]] = {
            "ATOM": [],
            "BOND": [],
            "IMPR": [],
            "ANGLES": [],
            "DIHE": [],
            "MASS": [],
        }
        auto: set[str] = set()
        atom_dict: dict[str, int] = {}

        with open(path, "r", encoding="utf8") as _f:
            f = FileIter(_f, start=1, stripper=lambda i: i.partition("!")[0].strip())
            statement = "<UNKNOWN>"
            try:
                # Skip the top-most header until the CHARMM version has been reached
                i = "*"
                while i.startswith("*"):
                    i = next(f)
                version = tuple(int(j) for j in i.split())

                # Parse all MASS statements
                i = next(f)
                statement = "MASS"
                while i.startswith("MASS"):
                    dct["MASS"].append(tuple(i.split()[1:]))
                    i = next(f)

                # Find the first RESI statement
                while not i.startswith("RESI"):
                    if i.startswith("AUTO"):
                        auto.update(i.split()[1:])
                    i = next(f)
                statement = "RESI"

                # Keep parsing all REST-related statements until END has been reached
                res_index = 1
                while i != "END":
                    # RESI-statements are not guaranteed to contain a residue name
                    res_fields = i.split()
                    if len(res_fields) == 2:
                        molecule = f"RES{res_index}"
                    else:
                        molecule = res_fields[1]
                    res_index += 1

                    j = 0
                    for i in f:
                        statement, *rest = i.split()
                        if statement == "RESI" or statement == "END":
                            break
                        lst = dct.get(statement)
                        if lst is not None:
                            if statement == "ATOM":
                                j += 1
                                atom_dict[rest[0]] = j
                                lst.append((molecule, j, *rest[1:]))
                            else:
                                lst.append((molecule, *(atom_dict[at] for at in rest)))
            except StopIteration as ex:
                raise ValueError(
                    f"{f.name!r}: failed to find a `END` statement at the end of the file"
                ) from ex
            except Exception as ex:
                raise ValueError(
                    f"{f.name!r}: failed to parse the {statement!r} statement on line {f.index!r}"
                ) from ex

        # Convert the lists into dataframes via a structured array intermediate
        # Numpy arrays have much better dtype control compared to pandas dataframes/series,
        # hence the array intermediate
        kwargs: dict[str, pd.DataFrame] = {}
        for k, v in dct.items():
            try:
                rec_array = np.fromiter(v, dtype=cls.DTYPES[k], count=len(v))
            except Exception as ex:
                msg = cls._get_err_msg(k, v)
                if msg is None:
                    raise
                else:
                    raise ValueError(f"{f.name!r}: {msg}") from ex
            df = pd.DataFrame(rec_array)
            df.set_index("molecule" if k != "MASS" else "index", drop=True, inplace=True)
            kwargs[k.lower()] = df
        return cls(charmm_version=version, auto=auto, **kwargs)

    def concatenate(self, rtf_iter: Iterable[RTFContainer]) -> Self:
        """Concatenate multiple RTFContainers into a single instance.

        Parameters
        ----------
        prm_iter : list[FOX.RTFContainer]
            A list with other RTFContainers to concatenate

        Returns
        -------
        FOX.PRMContainer
            The new concatenated RTFContainer

        """
        rtf_list: list[RTFContainer] = []
        for rtf in rtf_iter:
            if not isinstance(rtf, RTFContainer):
                raise TypeError("Expected a RTFContainer")
            rtf.auto_to_explicit()
            rtf_list.append(rtf)

        dct = {
            "mass": pd.concat([self.mass] + [rtf.mass for rtf in rtf_list], ignore_index=True),
            "atom": pd.concat([self.atom] + [rtf.atom for rtf in rtf_list]),
            "bond": pd.concat([self.bond] + [rtf.bond for rtf in rtf_list]),
            "impr": pd.concat([self.impropers] + [rtf.impropers for rtf in rtf_list]),
            "angles": pd.concat([self.angles] + [rtf.angles for rtf in rtf_list]),
            "dihe": pd.concat([self.dihedrals] + [rtf.dihedrals for rtf in rtf_list]),
        }
        dct["mass"].drop_duplicates("atom_type", inplace=True, ignore_index=True)

        cls = type(self)
        return cls(**dct)
