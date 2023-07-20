"""A class for reading and GROMACS .top topology files.

Index
-----
.. currentmodule:: FOX
.. autosummary::
    TOPContainer
    TOPContainer.from_file
    TOPContainer.to_file
    TOPContainer.allclose
    TOPContainer.generate_pairs
    TOPContainer.generate_pairs_nb
    TOPContainer.copy
    TOPContainer.concatenate

API
---
.. autoclass:: TOPContainer
    :noindex:
    :members: defaults, atomtypes, pairtypes, bondtypes, angletypes, dihedraltypes,
        constrainttypes, nonbond_params, moleculetype, atoms, pairs, bonds, angles,
        dihedrals, system, molecules, DF_DTYPES, DF_DICT_DTYPES

.. automethod:: TOPContainer.from_file
.. automethod:: TOPContainer.to_file
.. automethod:: TOPContainer.allclose
.. automethod:: TOPContainer.generate_pairs
.. automethod:: TOPContainer.generate_pairs_nb
.. automethod:: TOPContainer.copy
.. autoattribute:: TOPContainer.concatenate

"""

from __future__ import annotations

import os
import copy
import types
import textwrap
import pprint
import warnings
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, Literal, cast

import h5py
import numpy as np
import pandas as pd

from . import FileIter
from ._top_concat import _TOPConcat
from ..ff import degree_of_separation

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import NDArray

    _T = TypeVar("_T")
    _DFType = TypeVar("_DFType", bound=pd.DataFrame | dict[int, pd.DataFrame])

    _DFNames = Literal[
        "defaults",
        "atomtypes",
        "moleculetype",
        "atoms",
        "system",
        "molecules",
        "pairs",
        "pairs_nb",
        "bonds",
        "angles",
        "dihedrals",
    ]
    _DFDictNames = Literal[
        "bondtypes",
        "pairtypes",
        "angletypes",
        "dihedraltypes",
        "constrainttypes",
        "nonbond_params",
    ]
    _DirectiveNames = _DFNames | _DFDictNames

    _DtypeMap = types.MappingProxyType[_T, np.dtype[np.void]]
    _DFKwargs = dict[_DFNames, pd.DataFrame]
    _DFDictKwargs = defaultdict[_DFDictNames, dict[int, pd.DataFrame]]

__all__ = ["TOPContainer", "TOPDirectiveWarning"]


class TOPDirectiveWarning(Warning):
    """Class for warnings related to .top directives."""


def _df_allclose(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    rtol: float,
    atol: float,
    equal_nan: bool,
) -> bool:
    if (
        df1.shape != df2.shape
        or not np.all(df1.columns == df2.columns)
        or not np.all(df1.index == df2.index)
    ):
        return False

    for k, series1 in df1.items():
        series2 = df2[k]
        if np.issubdtype(series1.dtype, np.inexact):
            if not np.allclose(series1, series2, rtol, atol, equal_nan):
                return False
        elif not np.array_equal(series1, series2):
            return False
    return True


_DF_DICT_DTYPES: _DtypeMap[str] = types.MappingProxyType({
    "pairtypes_1": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("sigma", "f8"),
        ("epsilon", "f8"),
    ]),
    "pairtypes_2": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("fudgeQQ", "f8"),
        ("qi", "f8"),
        ("qj", "f8"),
        ("sigma", "f8"),
        ("epsilon", "f8"),
    ]),
    "bondtypes_1_2_6_7": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("b0", "f8"),
        ("k", "f8"),
    ]),
    "bondtypes_3": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("b0", "f8"),
        ("D", "f8"),
        ("beta", "f8"),
    ]),
    "bondtypes_4": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("b0", "f8"),
        ("C", "f8"),
    ]),
    "bondtypes_5": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
    ]),
    "bondtypes_8_9": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("table_num", "i8"),
        ("k", "f8"),
    ]),
    "bondtypes_10": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("low", "f8"),
        ("up", "f8"),
        ("k", "f8"),
    ]),
    "angletypes_1": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("func", "i8"),
        ("theta", "f8"),
        ("ktheta", "f8"),
        ("ub0", "f8"),
        ("kub", "f8"),
    ]),
    "angletypes_1_2_10": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("func", "i8"),
        ("theta", "f8"),
        ("k", "f8"),
    ]),
    "angletypes_3": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("func", "i8"),
        ("r1", "f8"),
        ("r2", "f8"),
        ("k", "f8"),
    ]),
    "angletypes_4": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("func", "i8"),
        ("r1", "f8"),
        ("r2", "f8"),
        ("r3", "f8"),
        ("k", "f8"),
    ]),
    "angletypes_5": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("func", "i8"),
        ("theta", "f8"),
        ("ktheta", "f8"),
        ("ub0", "f8"),
        ("kub", "f8"),
    ]),
    "angletypes_6": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("func", "i8"),
        ("theta", "f8"),
        ("C", "f8"),
    ]),
    "angletypes_8": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("func", "i8"),
        ("table_num", "i8"),
        ("k", "f8"),
    ]),
    "angletypes_9": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("func", "i8"),
        ("a", "f8"),
        ("k", "f8"),
    ]),
    "dihedraltypes_1_4_9": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("atom4", "U5"),
        ("func", "i8"),
        ("phi0", "f8"),
        ("k", "f8"),
        ("n", "i8"),
    ]),
    "dihedraltypes_2": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("atom4", "U5"),
        ("func", "i8"),
        ("xi0", "f8"),
        ("k", "f8"),
    ]),
    "dihedraltypes_3": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("atom4", "U5"),
        ("func", "i8"),
        ("C0", "f8"),
        ("C1", "f8"),
        ("C2", "f8"),
        ("C3", "f8"),
        ("C4", "f8"),
        ("C5", "f8"),
    ]),
    "dihedraltypes_5": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("atom4", "U5"),
        ("func", "i8"),
        ("C1", "f8"),
        ("C2", "f8"),
        ("C3", "f8"),
        ("C4", "f8"),
        ("C5", "f8"),
    ]),
    "dihedraltypes_8": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("atom4", "U5"),
        ("func", "i8"),
        ("table_num", "i8"),
        ("k", "f8"),
    ]),
    "dihedraltypes_10": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("atom4", "U5"),
        ("func", "i8"),
        ("phi0", "f8"),
        ("k", "f8"),
    ]),
    "dihedraltypes_11": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("atom3", "U5"),
        ("atom4", "U5"),
        ("func", "i8"),
        ("k", "f8"),
        ("a0", "f8"),
        ("a1", "f8"),
        ("a2", "f8"),
        ("a3", "f8"),
    ]),
    "constraints_1_2": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
    ]),
    "nonbond_params_1": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("sigma", "f8"),
        ("epsilon", "f8"),
    ]),
    "nonbond_params_2": np.dtype([
        ("atom1", "U5"),
        ("atom2", "U5"),
        ("func", "i8"),
        ("a", "f8"),
        ("b", "f8"),
        ("c", "f8"),
    ]),
})


class TOPContainer:
    """A class for managing GROMACS .top topology files.

    Examples
    --------
    .. code:: python

        >>> from FOX import TOPContainer

        >>> input_file: str = ...
        >>> output_file: str = ...

        >>> top = TOPContainer.from_file(input_file)
        >>> top.to_file(output_file)

    """

    __slots__ = (
        "__weakref__",
        "defaults",
        "atomtypes",
        "pairtypes",
        "bondtypes",
        "angletypes",
        "dihedraltypes",
        "constrainttypes",
        "nonbond_params",
        "moleculetype",
        "atoms",
        "pairs",
        "pairs_nb",
        "bonds",
        "angles",
        "dihedrals",
        "system",
        "molecules",
        "_concatenate",
    )

    # parameter level
    #: A dataframe holding the ``defaults`` directive.
    defaults: pd.DataFrame
    #: A dataframe holding the ``atomtypes`` directive.
    atomtypes: pd.DataFrame
    #: A dictionary of dataframes holding the ``bondtypes`` directive.
    bondtypes: dict[int, pd.DataFrame]
    #: A dictionary of dataframes holding the ``pairtypes`` directive.
    pairtypes: dict[int, pd.DataFrame]
    #: A dictionary of dataframes holding the ``angletypes`` directive.
    angletypes: dict[int, pd.DataFrame]
    #: A dictionary of dataframes holding the ``dihedraltypes`` directive.
    dihedraltypes: dict[int, pd.DataFrame]
    #: A dictionary of dataframes holding the ``constrainttypes`` directive.
    constrainttypes: dict[int, pd.DataFrame]
    #: A dictionary of dataframes holding the ``nonbond_params`` directive.
    nonbond_params: dict[int, pd.DataFrame]

    # molecule level
    #: A dataframe holding the ``moleculetype`` directive.
    moleculetype: pd.DataFrame
    #: A dataframe holding the ``atoms`` directive.
    atoms: pd.DataFrame
    #: A dataframe holding the ``pairs`` directive.
    pairs: pd.DataFrame
    #: A dataframe holding the ``pairs_nb`` directive.
    pairs_nb: pd.DataFrame
    #: A dataframe holding the ``bonds`` directive.
    bonds: pd.DataFrame
    #: A dataframe holding the ``angles`` directive.
    angles: pd.DataFrame
    #: A dataframe holding the ``dihedrals`` directive.
    dihedrals: pd.DataFrame

    # system level
    #: A dataframe holding the ``system`` directive.
    system: pd.DataFrame
    #: A dataframe holding the ``molecules`` directive.
    molecules: pd.DataFrame

    @property
    def concatenate(self) -> _TOPConcat:
        """Namespace with functions for adding new directive-specific rows.

        .. currentmodule:: FOX.io._top_concat._TOPConcat
        .. autofunction:: atomtypes
        .. autofunction:: nonbond_params
        .. autofunction:: atoms
        .. autofunction:: pairs
        .. autofunction:: pairs_nb

        """
        return self._concatenate

    #: A mapping holding the data types of all mandatory directives.
    DF_DTYPES: ClassVar[_DtypeMap[_DFNames]] = types.MappingProxyType({
        "defaults": np.dtype([
            ("nbfunc", "i8"),
            ("comb_rule", "i8"),
            ("gen_pairs", "U3"),
            ("fudgeLJ", "f8"),
            ("fudgeQQ", "f8"),
        ]),
        "atomtypes": np.dtype([
            ("atom_type", "U5"),
            ("atnum", "i8"),
            ("mass", "f8"),
            ("charge", "f8"),
            ("particle_type", "U1"),
            ("sigma", "f8"),
            ("epsilon", "f8"),
        ]),
        "moleculetype": np.dtype([
            ("molecule", "O"),
            ("n_rexcl", "i8"),
        ]),
        "atoms": np.dtype([
            ("molecule", "O"),
            ("atom1", "i8"),
            ("atom_type", "U5"),
            ("res_num", "i8"),
            ("res_name", "U5"),
            ("atom_name", "U5"),
            ("charge_group", "i8"),
            ("charge", "f8"),
            ("mass", "f8"),
        ]),
        "system": np.dtype([
            ("name", "O"),
        ]),
        "molecules": np.dtype([
            ("molecule", "O"),
            ("n_mol", "i8"),
        ]),
        "bonds": np.dtype([
            ("molecule", "O"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("func", "i8"),
        ]),
        "angles": np.dtype([
            ("molecule", "O"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("atom3", "i8"),
            ("func", "i8"),
        ]),
        "dihedrals": np.dtype([
            ("molecule", "O"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("atom3", "i8"),
            ("atom4", "i8"),
            ("func", "i8"),
        ]),
        "pairs": np.dtype([
            ("molecule", "O"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("func", "i8"),
        ]),
        "pairs_nb": np.dtype([
            ("molecule", "O"),
            ("atom1", "i8"),
            ("atom2", "i8"),
            ("func", "i8"),
        ]),
    })

    #: A mapping holding the data types of all optional (dictionary of dataframe based) directives.
    DF_DICT_DTYPES: ClassVar[
        types.MappingProxyType[_DFDictNames, _DtypeMap[int]]
    ] = types.MappingProxyType({
        "pairtypes": types.MappingProxyType({
            1: _DF_DICT_DTYPES["pairtypes_1"],
            2: _DF_DICT_DTYPES["pairtypes_2"],
        }),
        "bondtypes": types.MappingProxyType({
            1: _DF_DICT_DTYPES["bondtypes_1_2_6_7"],
            2: _DF_DICT_DTYPES["bondtypes_1_2_6_7"],
            3: _DF_DICT_DTYPES["bondtypes_3"],
            4: _DF_DICT_DTYPES["bondtypes_4"],
            5: _DF_DICT_DTYPES["bondtypes_5"],
            6: _DF_DICT_DTYPES["bondtypes_1_2_6_7"],
            7: _DF_DICT_DTYPES["bondtypes_1_2_6_7"],
            8: _DF_DICT_DTYPES["bondtypes_8_9"],
            9: _DF_DICT_DTYPES["bondtypes_8_9"],
            10: _DF_DICT_DTYPES["bondtypes_10"],
        }),
        "angletypes": types.MappingProxyType({
            1: _DF_DICT_DTYPES["angletypes_1_2_10"],
            2: _DF_DICT_DTYPES["angletypes_1_2_10"],
            3: _DF_DICT_DTYPES["angletypes_3"],
            4: _DF_DICT_DTYPES["angletypes_4"],
            5: _DF_DICT_DTYPES["angletypes_5"],
            6: _DF_DICT_DTYPES["angletypes_6"],
            8: _DF_DICT_DTYPES["angletypes_8"],
            9: _DF_DICT_DTYPES["angletypes_9"],
            10: _DF_DICT_DTYPES["angletypes_1_2_10"],
        }),
        "dihedraltypes": types.MappingProxyType({
            1: _DF_DICT_DTYPES["dihedraltypes_1_4_9"],
            2: _DF_DICT_DTYPES["dihedraltypes_2"],
            3: _DF_DICT_DTYPES["dihedraltypes_3"],
            4: _DF_DICT_DTYPES["dihedraltypes_1_4_9"],
            5: _DF_DICT_DTYPES["dihedraltypes_5"],
            8: _DF_DICT_DTYPES["dihedraltypes_8"],
            9: _DF_DICT_DTYPES["dihedraltypes_1_4_9"],
            10: _DF_DICT_DTYPES["dihedraltypes_10"],
            11: _DF_DICT_DTYPES["dihedraltypes_11"],
        }),
        "constrainttypes": types.MappingProxyType({
            1: _DF_DICT_DTYPES["constraints_1_2"],
            2: _DF_DICT_DTYPES["constraints_1_2"],
        }),
        "nonbond_params": types.MappingProxyType({
            1: _DF_DICT_DTYPES["nonbond_params_1"],
            2: _DF_DICT_DTYPES["nonbond_params_2"],
        }),
    })

    def __init__(
        self,
        *,
        defaults: None | pd.DataFrame = None,
        atomtypes: None | pd.DataFrame = None,
        moleculetype: None | pd.DataFrame = None,
        atoms: None | pd.DataFrame = None,
        system: None | pd.DataFrame = None,
        molecules: None | pd.DataFrame = None,
        bondtypes: None | dict[int, pd.DataFrame] = None,
        pairtypes: None | dict[int, pd.DataFrame] = None,
        angletypes: None | dict[int, pd.DataFrame] = None,
        dihedraltypes: None | dict[int, pd.DataFrame] = None,
        constrainttypes: None | dict[int, pd.DataFrame] = None,
        nonbond_params: None | dict[int, pd.DataFrame] = None,
        pairs: None | pd.DataFrame = None,
        pairs_nb: None | pd.DataFrame = None,
        bonds: None | pd.DataFrame = None,
        angles: None | pd.DataFrame = None,
        dihedrals: None | pd.DataFrame = None,
    ) -> None:
        """Initialize the instance."""
        self.defaults = self._validate_attr(defaults, "defaults")
        self.atomtypes = self._validate_attr(atomtypes, "atomtypes")
        self.bondtypes = self._validate_attr(bondtypes, "bondtypes")
        self.pairtypes = self._validate_attr(pairtypes, "pairtypes")
        self.angletypes = self._validate_attr(angletypes, "angletypes")
        self.dihedraltypes = self._validate_attr(dihedraltypes, "dihedraltypes")
        self.constrainttypes = self._validate_attr(constrainttypes, "constrainttypes")
        self.nonbond_params = self._validate_attr(nonbond_params, "nonbond_params")

        self.moleculetype = self._validate_attr(moleculetype, "moleculetype")
        self.atoms = self._validate_attr(atoms, "atoms")
        self.pairs = self._validate_attr(pairs, "pairs")
        self.pairs_nb = self._validate_attr(pairs_nb, "pairs_nb")
        self.bonds = self._validate_attr(bonds, "bonds")
        self.angles = self._validate_attr(angles, "angles")
        self.dihedrals = self._validate_attr(dihedrals, "dihedrals")

        self.system = self._validate_attr(system, "system")
        self.molecules = self._validate_attr(molecules, "molecules")

        self._concatenate = _TOPConcat(self, "concatenate")

    @classmethod
    def _from_dict(cls, dct: dict[_DFNames | _DFDictNames, Any]) -> Self:
        """Helper function for :meth:`TOPContainer.__reduce__`."""
        return cls(**dct)

    @staticmethod
    def _validate_df(
        attr: pd.DataFrame,
        dtype: np.dtype[np.void],
        name: _DirectiveNames,
        func: int | None = None,
    ) -> None:
        assert dtype.names is not None
        expected = set(dtype.names)
        observed = set(attr.columns)
        if expected != observed:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            name_repr = repr(name) if func is None else f"{name!r} (func {func})"
            raise ValueError(
                f"Key mismatch in argument {name_repr}:\n"
                f"    missing keys: {missing!r}\n"
                f"    extra keys: {extra!r}"
            )

    @classmethod
    def _validate_attr(cls, attr: None | _DFType, name: _DirectiveNames) -> _DFType:
        """Perform some basic validation on the passed dataframe."""
        if (dtype := cls.DF_DTYPES.get(name)) is not None:
            if attr is None:
                return pd.DataFrame.from_records(np.empty((0,), dtype=dtype))
            elif not isinstance(attr, pd.DataFrame):
                raise TypeError(
                    f"Argument {name!r} expects None or a dataframe; "
                    f"observed type: {type(attr).__module__}.{type(attr).__name__}"
                )
            else:
                cls._validate_df(attr, dtype, name)

        elif (dtype_dict := cls.DF_DICT_DTYPES.get(name)) is not None:
            if attr is None:
                return {}
            elif (
                not isinstance(attr, dict)
                or not all(isinstance(v, pd.DataFrame) for v in attr.values())
            ):
                raise TypeError(
                    f"Argument {name!r} expects None or a a dictionary of dataframes; "
                    f"observed type: {type(attr).__module__}.{type(attr).__name__}")
            else:
                for func, df in attr.items():
                    try:
                        dtype = dtype_dict[func]
                    except KeyError:
                        raise ValueError(f"Invalid {name!r} func type: {func!r}") from None
                    cls._validate_df(df, dtype, name, func)

        else:
            raise ValueError(f"Unknown directive {name!r}")
        return attr

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[Any, ...]]:
        """Helper function for :mod:`pickle`."""
        cls = type(self)
        return cls._from_dict, ({
            "defaults": self.defaults,
            "atomtypes": self.atomtypes,
            "bondtypes": self.bondtypes,
            "pairtypes": self.pairtypes,
            "angletypes": self.angletypes,
            "dihedraltypes": self.dihedraltypes,
            "constrainttypes": self.constrainttypes,
            "nonbond_params": self.nonbond_params,
            "moleculetype": self.moleculetype,
            "atoms": self.atoms,
            "pairs": self.pairs,
            "pairs_nb": self.pairs_nb,
            "bonds": self.bonds,
            "angles": self.angles,
            "dihedrals": self.dihedrals,
            "system": self.system,
            "molecules": self.molecules,
        },)

    def __repr__(self) -> str:
        """Implement :func:`repr(self)<repr>`."""
        # Get all to-be printed attribute (names)
        cls = type(self)
        attr_names = [
            "defaults",
            "atomtypes",
            "pairtypes",
            "bondtypes",
            "angletypes",
            "dihedraltypes",
            "constrainttypes",
            "nonbond_params",
            "moleculetype",
            "atoms",
            "pairs",
            "pairs_nb",
            "bonds",
            "angles",
            "dihedrals",
            "system",
            "molecules",
        ]

        # Determine the indentation width
        width = max(len(k) for k in attr_names)
        indent = width + 3

        # Gather string representations of all attributes
        ret = ""
        with pd.option_context("display.max_rows", 20):
            items = ((k, getattr(self, k)) for k in attr_names)
            for k, _v in items:
                v = textwrap.indent(
                    pprint.pformat(_v, sort_dicts=False, width=100),
                    " " * indent,
                )[indent:]
                ret += f"{k:{width}} = {v},\n"
        return f"{cls.__name__}(\n{textwrap.indent(ret[:-2], 4 * ' ')}\n)"

    def __eq__(self, other: object) -> bool:
        """Implement :meth:`self == other<object.__eq__>`."""
        cls = type(self)
        if not isinstance(other, cls):
            return NotImplemented
        return self._compare(other, lambda i, j: i.equals(j))

    def allclose(
        self,
        other: TOPContainer,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = True,
    ) -> bool:
        """Return whether two TOPContainers are equivalent within a given tolerance.

        Parameters
        ----------
        other: TOPContainer
            The to-be compared TOPContainer
        rtol: float
            The relative tolerance parameter (see Notes).
        atol: float
            The absolute tolerance parameter (see Notes).
        equal_nan: bool
            Whether to compare NaN's as equal.
            If True, NaN's in a will be considered equal to NaN's in b in the output array.

        Returns
        -------
        bool
            Whether the two containers are equivalent within a given tolerance.

        See Also
        --------
        :func:`numpy.allclose`
            Returns True if two arrays are element-wise equal within a tolerance.

        """
        cls = type(self)
        if not isinstance(other, cls):
            raise TypeError(f"Expected a TOPContainer; observed type: {type(other).__name__!r}")
        return self._compare(other, lambda i, j: _df_allclose(i, j, rtol, atol, equal_nan))

    def _compare(
        self,
        other: TOPContainer,
        callback: Callable[[pd.Series, pd.Series], bool],
    ) -> bool:
        for name1 in self.DF_DTYPES:
            df1: pd.DataFrame = getattr(self, name1)
            df2: pd.DataFrame = getattr(other, name1)
            if not callback(df1, df2):
                return False

        for name2 in self.DF_DICT_DTYPES:
            df_dict1: dict[int, pd.DataFrame] = getattr(self, name2)
            df_dict2: dict[int, pd.DataFrame] = getattr(other, name2)
            if df_dict1.keys() != df_dict2.keys():
                return False
            elif not all(callback(v, df_dict2[k]) for k, v in df_dict1.items()):
                return False
        return True

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> Self:
        """Construct a new :class:`TOPContainer` from the passed file path.

        Parameters
        ----------
        path : path-like object
            The path to the .top file

        Returns
        -------
        FOX.TOPContainer
            A newly constructed .top container

        """
        df_kwargs: _DFKwargs = {}
        df_dict_kwargs: _DFDictKwargs = defaultdict(dict)
        requires_mol = {"atoms", "pairs", "pairs_nb", "bonds", "angles", "dihedrals"}
        mol: str | None = None

        with open(path, "r", encoding="utf8") as _f:
            f = FileIter(_f, stripper=lambda i: i.partition(";")[0].strip())
            directive: None | _DirectiveNames = None
            for i in f:
                while not (i.startswith("[") and i.endswith("]")):
                    next(f)
                else:
                    directive = cast("_DirectiveNames", i.strip("[] "))
                    break

            while directive is not None:
                old_directive = directive
                if old_directive in requires_mol:
                    directive, func, df, mol_new = cls._parse_directive(f, old_directive, mol)
                else:
                    directive, func, df, mol_new = cls._parse_directive(f, old_directive)

                if mol_new is not None:
                    mol = mol_new
                if df is None:
                    continue

                if old_directive in requires_mol or old_directive == "moleculetype":
                    if old_directive in df_kwargs:
                        df_kwargs[old_directive] = pd.concat(
                            [df_kwargs[old_directive], df], copy=False,
                        )
                    else:
                        df_kwargs[old_directive] = df
                elif old_directive in cls.DF_DTYPES:
                    df_kwargs[old_directive] = df
                elif old_directive in cls.DF_DICT_DTYPES:
                    df_dict_kwargs[old_directive][func] = df
        return cls(**df_kwargs, **df_dict_kwargs)

    @staticmethod
    def _get_func_index(dtype_dict: _DtypeMap[int]) -> int:
        dtype = dtype_dict[1]
        assert dtype.names is not None
        return dtype.names.index("func")

    @staticmethod
    def _get_err_msg(
        start: int | None,
        end: int | None,
        directive: _DirectiveNames,
        dtype: np.dtype[np.void],
        lst: list[tuple[str, ...]],
    ) -> str:
        if start is None or end is None:
            return f'failed to parse one or more entries in the "[ {directive} ]"'

        for i, tup in enumerate(lst, start=1):
            try:
                np.array(tup, dtype=dtype)
            except Exception:
                return (
                    f'failed to parse "[ {directive} ]" directive entry {i} '
                    f"between lines {start + 1} and {end - 1}"
                )
        return (
            f'failed to parse the "[ {directive} ]" directive between '
            f'lines {start + 1} and {end - 1}'
        )

    @classmethod
    def _parse_directive(
        cls,
        f: FileIter,
        directive: _DirectiveNames,
        prefix: None | str = None,
    ) -> tuple[None | _DirectiveNames, int, None | pd.DataFrame, None | str]:
        # Get all data within the current directive
        lst: list[tuple[str, ...]] = []
        i_start = f.index
        mol_new: str | None = None
        try:
            while not ((item := next(f)).startswith("[") and item.endswith("]")):
                lst.append(tuple(item.split()) if prefix is None else (prefix, *item.split()))
        except StopIteration:
            new_directive = None
        else:
            new_directive = cast("_DirectiveNames", item.strip("[] "))

        # Get the data type of the current directive
        func = 1
        if (dtype := cls.DF_DTYPES.get(directive)) is not None:
            pass
        elif (dtype_dict := cls.DF_DICT_DTYPES.get(directive)) is not None:
            if len(lst) == 0:
                return (new_directive, func, None, mol_new)
            func = int(lst[0][cls._get_func_index(dtype_dict)])
            dtype = dtype_dict[func]
        else:
            warnings.warn(
                f'Invalid or unsuported directive: "[ {directive} ]"',
                TOPDirectiveWarning, stacklevel=3,
            )
            return (new_directive, func, None, mol_new)

        # Convert the data into a dataframe
        try:
            rec_array = np.array(lst, dtype=dtype)
        except Exception as ex:
            msg = cls._get_err_msg(i_start, f.index, directive, dtype, lst)
            raise ValueError(f"{f.name!r}: {msg}") from ex
        df = pd.DataFrame(rec_array)

        if directive == "moleculetype":
            assert len(df) == 1
            mol_new = df.at[0, "molecule"]
        return (new_directive, func, df, mol_new)

    def to_file(self, path: str | os.PathLike[str]) -> None:
        """Construct a new .top file from this instance.

        Parameters
        ----------
        path : path-like object
            The path of the to-be created .top file

        """
        kwargs = {"index": False, "col_space": 5}
        with open(path, "w", encoding="utf8") as f:
            # Parameter level
            f.write("[ defaults ]\n")
            if self.defaults.size:
                f.write(";")
                self.defaults.to_string(f, **kwargs)
                f.write("\n")

            f.write("\n[ atomtypes ]\n")
            if self.atomtypes.size:
                f.write(";")
                self.atomtypes.to_string(f, **kwargs)
                f.write("\n")

            for dir1 in self.DF_DICT_DTYPES:
                df_dict: Mapping[str, pd.DataFrame] = getattr(self, dir1)
                for df in df_dict.values():
                    if df.size:
                        f.write(f"\n[ {dir1} ]\n")
                        f.write(";")
                        df.to_string(f, **kwargs)
                        f.write("\n")

            # Molecule level
            mol_directives = [
                "moleculetype",
                "atoms",
                "pairs",
                "pairs_nb",
                "bonds",
                "angles",
                "dihedrals"
            ]
            for name in self.moleculetype["molecule"]:
                for dir2 in mol_directives:
                    if (df := getattr(self, dir2)) is None:
                        continue
                    if dir2 == "moleculetype":
                        columns: slice | list[str] = slice(None)
                    else:
                        columns = [k for k in df if k != "molecule"]

                    df = df.loc[df["molecule"] == name, columns]
                    if df.size:
                        f.write(f"\n[ {dir2} ]\n")
                        f.write(";")
                        df.to_string(f, **kwargs)
                        f.write("\n")

            # System level
            f.write("\n[ system ]\n")
            if self.system.size:
                f.write(";")
                self.system.to_string(f, **kwargs)
                f.write("\n")

            f.write("\n[ molecules ]\n")
            if self.molecules.size:
                f.write(";")
                self.molecules.to_string(f, **kwargs)
                f.write("\n")

    def _to_hdf5_dict(self) -> dict[str, NDArray[np.void]]:
        dtype_dct: dict[str, NDArray[np.void]] = {}
        for name1, _dtype in self.DF_DTYPES.items():
            df: pd.DataFrame = getattr(self, name1, None)
            assert _dtype.fields is not None

            # Construct a h5py-compatible structured dtype
            dtype_list = []
            for sub_field, (sub_dtype, *_) in _dtype.fields.items():
                if sub_dtype.kind == "U":
                    sub_dtype = h5py.string_dtype("utf-8", sub_dtype.itemsize // 4)
                elif sub_dtype.kind == "O":
                    sub_dtype = h5py.string_dtype("utf-8")
                dtype_list.append((sub_field, sub_dtype))
            dtype = np.dtype(dtype_list)
            dtype_dct[name1] = df.to_records(index=False).astype(dtype)

        iterator = (
            (k, i, _dtype) for k, dct in self.DF_DICT_DTYPES.items() for i, _dtype in dct.items()
        )
        for (name2, func, _dtype) in iterator:
            df = (pre_df.get(func) if (pre_df := getattr(self, name2)) is not None else None)
            if df is None:
                continue
            assert _dtype.fields is not None

            # Construct a h5py-compatible structured dtype
            dtype_list = []
            for sub_field, (sub_dtype, *_) in _dtype.fields.items():
                if sub_dtype.kind == "U":
                    sub_dtype = h5py.string_dtype("utf-8", sub_dtype.itemsize // 4)
                elif sub_dtype.kind == "O":
                    sub_dtype = h5py.string_dtype("utf-8")
                dtype_list.append((sub_field, sub_dtype))
            dtype = np.dtype(dtype_list)
            dtype_dct[f"{name2}/{func}"] = df.to_records(index=False).astype(dtype)
        return dtype_dct

    def generate_pairs(self, func: Literal[1, 2] = 1) -> None:
        """Construct and populate the ``pairs`` directive with explicit 1,4-pairs based on \
        the available bonds.

        Parameters
        ----------
        func: {1, 2}
            The func type as used for the new pairs.

        """
        pair_dfs: list[pd.DataFrame] = []
        for mol in self.molecules["molecule"]:
            atom_count = len(self.atoms.loc[self.atoms["molecule"] == mol, :])
            bonds = self.bonds.loc[self.bonds["molecule"] == mol, ["atom1", "atom2"]] - 1
            if self.bonds.size == 0:
                continue

            depth_mat = np.triu(degree_of_separation(
                atom_count * [None],
                bond_mat=(np.ones(len(bonds), dtype=np.bool_), (bonds["atom1"], bonds["atom2"]))
            ))
            pairs_14 = np.array(np.where(depth_mat == 3), dtype=self.DF_DTYPES["pairs"]["atom1"])
            pairs_14 += 1
            pair_dfs.append(pd.DataFrame({
                "molecule": mol,
                "atom1": pairs_14[0],
                "atom2": pairs_14[1],
                "func": func,
            }))
        if len(pair_dfs) == 0:
            return

        keys = ["molecule", "atom1", "atom2"]
        pairs_new = pd.concat(pair_dfs, ignore_index=True)
        self.pairs = pairs_new[~pairs_new.duplicated(keys)].sort_values(keys, ignore_index=True)

    def generate_pairs_nb(self, func: Literal[1] = 1) -> None:
        """Construct and populate the ``pairs_nb`` directive with explicit nonbonded pairs based \
        on the available non-bonded atoms.

        Parameters
        ----------
        func: {1}
            The func type as used for the new pairs.

        """
        pair_dfs: list[pd.DataFrame] = []
        for mol in self.molecules["molecule"]:
            atom_count = len(self.atoms.loc[self.atoms["molecule"] == mol, :])
            bonds = self.bonds.loc[self.bonds["molecule"] == mol, ["atom1", "atom2"]] - 1
            if self.bonds.size == 0:
                continue

            depth_mat = np.triu(degree_of_separation(
                atom_count * [None],
                bond_mat=(np.ones(len(bonds), dtype=np.bool_), (bonds["atom1"], bonds["atom2"]))
            ))
            pairs = np.asarray(
                np.where(np.isinf(depth_mat)),
                dtype=self.DF_DTYPES["pairs_nb"]["atom1"],
            )
            pairs += 1
            pair_dfs.append(pd.DataFrame({
                "molecule": mol,
                "atom1": pairs[0],
                "atom2": pairs[1],
                "func": func,
            }))
        if len(pair_dfs) == 0:
            return

        keys = ["molecule", "atom1", "atom2"]
        pairs_new = pd.concat(pair_dfs, ignore_index=True)
        self.pairs_nb = pairs_new[~pairs_new.duplicated(keys)].sort_values(keys, ignore_index=True)

    def copy(self, deep: bool = True) -> Self:
        """Return a copy of this instance.

        Parameters
        ----------
        deep: bool
            Whether a deep copy should be created or not

        Returns
        -------
        A copy of this instance

        """
        return copy.deepcopy(self) if deep else copy.copy(self)
