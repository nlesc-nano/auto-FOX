"""A class for reading and CHARMM .rtf topology files.

Index
-----
.. currentmodule:: FOX
.. autosummary::
    RTFContainer
    RTFContainer.collapse_charges
    RTFContainer.auto_to_explicit
    RTFContainer.from_file

API
---
.. autoclass:: RTFContainer
    :noindex:
    :members: mass, atom, bond, impropers, angles, dihedrals, charmm_version, auto

.. automethod:: RTFContainer.collapse_charges
.. automethod:: RTFContainer.auto_to_explicit
.. automethod:: RTFContainer.from_file

"""
from __future__ import annotations

import os
import sys
import types
import textwrap
import itertools
import warnings
from typing import TYPE_CHECKING, Any, ClassVar, Literal
from collections.abc import Mapping
from collections import defaultdict

import numpy as np
import pandas as pd
from scm.plams import Molecule, Atom

from . import FileIter
from ..functions.molecule_utils import get_angles, get_dihedrals

if sys.version_info >= (3, 9):
    from collections.abc import Iterator
else:
    from typing import Iterator

if TYPE_CHECKING:
    from typing_extensions import Self

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

    #: A mapping with strucutred dtypes for each dataframe column and index
    DTYPES: ClassVar[types.MappingProxyType[str, np.dtype[Any]]] = types.MappingProxyType({
        "MASS": np.dtype([
            ("index", "i8"),
            ("atom_type", "U4"),
            ("mass", "f8"),
            ("atom_name", "U2"),
        ]),
        "ATOM": np.dtype([
            ("res_name", "U4"),
            ("atom1", "i8"),
            ("atom_type", "U4"),
            ("charge", "f8"),
        ]),
        "BOND": np.dtype([
            ("res_name", "U4"),
            ("atom1", "i8"),
            ("atom2", "i8"),
        ]),
        "ANGLES": np.dtype([
            ("res_name", "U4"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("atom3", "i8"),
        ]),
        "DIHE": np.dtype([
            ("res_name", "U4"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("atom3", "i8"),
            ("atom4", "i8"),
        ]),
        "IMPR": np.dtype([
            ("res_name", "U4"),
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
        ret = ''
        with pd.option_context(*self.pd_printoptions):
            items = ((k, getattr(self, k)) for k in attr_names)
            for k, _v in items:
                v = textwrap.indent(repr(_v), ' ' * indent)[indent:]
                ret += f'{k:{width}} = {v},\n'
            ret += f'{"auto":{width}} = {self.auto!r},\n'
            ret += f'{"charmm_version":{width}} = {self.charmm_version!r},\n'
        return f'{cls.__name__}(\n{textwrap.indent(ret[:-2], 4 * " ")}\n)'

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

        # Computer the angles/dihedrals for all molecules
        array_dict = {}
        for res, mol in mol_dict.items():
            array_dict[res] = func(mol)

        # Concatenate the residue-specific angles/dihedrals into a single structured array
        i = j = 0
        total_array = np.empty(sum(len(i) for i in array_dict.values()), dtype=dtype)
        for res, array in array_dict.items():
            j += len(array)
            total_array["res_name"][i:j] = res
            for k, field_name in enumerate(dtype.names[1:]):
                total_array[field_name][i:j] = array[..., k]
            i += len(array)

        # Convert the strucutred array into a dataframe
        df = pd.DataFrame(total_array)
        df.set_index("res_name", inplace=True, drop=True)
        return df

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Construct a new :class:`RTFContainer` from the passed file path.

        Parameter
        ---------
        path : :term:`python:path-like` object
            The path to the .rtf file

        Returns
        -------
        FOX.RTFContaier
            A newly constructed .rtf container

        """
        mass = []
        dct: dict[str, list[tuple[Any, ...]]] = {
            "ATOM": [],
            "BOND": [],
            "IMPR": [],
            "ANGLES": [],
            "DIHE": [],
        }
        auto = set()
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
                    mass.append(tuple(i.split()[1:]))
                    i = next(f)

                # Find the first RESI statement
                while not i.startswith("RESI"):
                    if i.startswith("AUTO"):
                        auto.update(i.split()[1:])
                    i = next(f)
                statement = "RESI"

                # Keep parsing all REST-related statements until END has been reached
                while i != "END":
                    _, res_name, _ = i.split()
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
                                lst.append((res_name, j, *rest[1:]))
                            else:
                                lst.append((res_name, *(atom_dict[at] for at in rest)))

            except StopIteration as ex:
                raise ValueError(
                    f"{f.name!r}: failed to find a `END` statement at the end of the file"
                ) from ex
            except Exception as ex:
                raise ValueError(
                    f"{f.name!r}: failed to parse the {statement!r} statement on line {f.index!r}"
                ) from ex

        kwargs: dict[str, pd.DataFrame] = {}
        for k, v in dct.items():
            df = pd.DataFrame(np.fromiter(v, dtype=cls.DTYPES[k], count=len(v)))
            df.set_index("res_name", drop=True, inplace=True)
            kwargs[k.lower()] = df
        kwargs["mass"] = pd.DataFrame(np.fromiter(mass, dtype=cls.DTYPES["MASS"], count=len(mass)))
        kwargs["mass"].set_index("index", inplace=True, drop=True)
        return cls(charmm_version=version, auto=auto, **kwargs)
