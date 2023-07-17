from __future__ import annotations

from collections.abc import Sequence, Callable
from typing import TYPE_CHECKING, Any, Literal, overload, TypeVar

import numpy as np
import pandas as pd
from scm.plams import PT

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import NDArray, ArrayLike

    from . import TOPContainer

    _SCT = TypeVar("_SCT", bound=np.generic)

    _ArrayLikeInt = int | Sequence[int] | NDArray[np.integer[Any]]
    _ArrayLikeFloat = float | Sequence[float] | NDArray[np.integer[Any]] | NDArray[np.floating[Any]]
    _ArrayLikeStr = str | Sequence[str] | NDArray[np.str_] | NDArray[np.object_]

__all__ = ["_TOPConcat"]


def _parse_int(arg: _ArrayLikeInt) -> NDArray[np.int64]:
    """Parse and validate int-based array-likes."""
    array = np.asarray(arg)
    if array.ndim > 1:
        raise ValueError("Expected a scalar or 1d array")
    return array.astype(np.int64, copy=False, casting="same_kind")


def _parse_float(arg: _ArrayLikeFloat) -> NDArray[np.float64]:
    """Parse and validate float-based array-likes."""
    array = np.asarray(arg)
    if array.ndim > 1:
        raise ValueError("Expected a scalar or 1d array")
    return array.astype(np.float64, copy=False, casting="same_kind")


def _parse_str(arg: _ArrayLikeStr, size: int) -> NDArray[np.str_]:
    """Parse and validate str-based array-likes."""
    array = np.asarray(arg)
    if array.ndim > 1:
        raise ValueError("Expected a scalar or 1d array")
    if array.dtype.type == np.object_:
        return array.astype(np.str_, copy=False).astype(
            (np.str_, size), casting="safe", copy=False
        )
    else:
        return array.astype((np.str_, size), casting="safe", copy=False)


def _parse_object(arg: ArrayLike) -> NDArray[np.object_]:
    """Parse and validate object-based array-likes."""
    array = np.asarray(arg)
    if array.ndim > 1:
        raise ValueError("Expected a scalar or 1d array")
    return array.astype(np.object_, copy=False)


def _parse(arg: ArrayLike, dtype: np.dtype[_SCT]) -> NDArray[_SCT]:
    """Wrapper for validating dtype-specific array-likes."""
    if dtype.type == np.float64:
        return _parse_float(arg)
    elif dtype.type == np.int64:
        return _parse_int(arg)
    elif dtype.type == np.str_:
        return _parse_str(arg, dtype.itemsize // 4)
    elif dtype.type == np.object_:
        return _parse_object(arg)
    else:
        raise ValueError(f"Unsupported dtype {dtype!r}")


class _TOPConcat:
    """A :class:`FOX.TOPContainer` helper for handling directive concatenation."""

    __slots__ = ("__weakref__", "__self__", "__dict__")

    if TYPE_CHECKING:
        @property
        def __self__(self) -> TOPContainer: ...
        @property
        def __name__(self) -> str: ...
        @property
        def __qualname__(self) -> str: ...

    def __init__(self, top: TOPContainer, name: str) -> None:
        """Initialize the instance."""
        super().__setattr__("__self__", top)
        super().__setattr__("__name__", name)
        super().__setattr__("__qualname__", f"{type(top).__qualname__}.{name}")

    def __setattr__(self, name: str, value: Any) -> None:
        """Implement :func:`setattr(self, name, value)<setattr>`."""
        if name == "__weakref__" or not hasattr(self, name):
            return super().__setattr__(name, value)
        else:
            raise AttributeError(f"{self.__qualname__!r} object attribute {name!r} is read-only")

    def __delattr__(self, name: str) -> None:
        """Implement :func:`delattr(self, name)<delattr>`."""
        if name == "__weakref__" or not hasattr(self, name):
            return super().__delattr__(name)
        else:
            raise AttributeError(f"{self.__qualname__!r} object attribute {name!r} is read-only")

    def __eq__(self, other: object) -> bool:
        """Implement :func:`self == other<object.__eq__>`."""
        cls = type(self)
        if not isinstance(other, cls):
            return NotImplemented
        else:
            return self.__self__ is other.__self__

    def __hash__(self) -> int:
        """Implement :func:`hash(self)<hash>`."""
        return id(self.__self__)

    def __repr__(self) -> str:
        """Implement :func:`repr(self)<repr>`."""
        return f"<bound {self.__name__!r} wrapper of {object.__repr__(self.__self__)}>"

    def __reduce__(self) -> tuple[Callable[..., Self], tuple[Any, ...]]:
        """Helper function for :mod:`pickle`."""
        return getattr, (self.__self__, self.__name__)

    @overload
    def atomtypes(
        self,
        *,
        atnum: _ArrayLikeInt,
        atom_type: None | _ArrayLikeStr = ...,
        charge: _ArrayLikeFloat = ...,
        sigma: _ArrayLikeFloat = ...,
        epsilon: _ArrayLikeFloat = ...,
        particle_type: Literal["A", "S", "V", "D"] | Sequence[Literal["A", "S", "V", "D"]] = ...,
    ) -> None: ...

    @overload
    def atomtypes(
        self,
        *,
        symbol: _ArrayLikeStr,
        atom_type: None | _ArrayLikeStr = ...,
        charge: _ArrayLikeFloat = ...,
        sigma: _ArrayLikeFloat = ...,
        epsilon: _ArrayLikeFloat = ...,
        particle_type: Literal["A", "S", "V", "D"] | Sequence[Literal["A", "S", "V", "D"]] = ...,
    ) -> None: ...

    def atomtypes(
        self,
        *,
        atnum: None | _ArrayLikeInt = None,
        symbol: None | _ArrayLikeStr = None,
        atom_type: None | _ArrayLikeStr = None,
        charge: _ArrayLikeFloat = 0.0,
        sigma: _ArrayLikeFloat = 0.0,
        epsilon: _ArrayLikeFloat = 0.0,
        particle_type: Literal["A", "S", "V", "D"] | Sequence[Literal["A", "S", "V", "D"]] = "A",
    ) -> None:
        """Add one or more atom types to the ``atomtypes`` directive.

        Examples
        --------
        .. code-block:: python

            >>> import FOX

            >>> top: FOX.TOPContainer = ...
            >>> top.concatenate.atomtypes(atnum=[6, 6, 7], sigma: [1.5, 1.2, 5.0], charge=0)

        Parameters
        ----------
        atnum/symbol : array-like
            One or more atomic numbers _or_ atomic symbols
        atom_type : array-like
            One or more atom types. If not provided use normal atomic symbols instead
        charge : array-like
            One or more atomic charges
        sigma : array-like
            One or more Lennard-Jones sigma values
        epsilon : array-like
            One or more Lennard-Jones epsilon values
        particle_type : {"A", "S", "V", "D"}
            One or more particule types

        """
        dtype = self.__self__.DF_DTYPES["atomtypes"]
        if atnum is not None:
            atnum = _parse(atnum, dtype["atnum"])
            symbol = np.fromiter((PT.get_symbol(i) for i in atnum.flat), dtype=dtype["atom_type"])
        elif symbol is not None:
            symbol = _parse(symbol, dtype["atom_type"])
            atnum = np.fromiter(
                (PT.get_atomic_number(i) for i in symbol.flat),
                dtype=dtype["atnum"],
            )
        else:
            raise TypeError("Either atnum or symbol must be specified")

        kwargs = {
            "atom_type": symbol if atom_type is None else _parse(atom_type, dtype["atom_type"]),
            "atnum": atnum,
            "mass": np.fromiter((PT.get_mass(i) for i in atnum.flat), dtype=dtype["mass"]),
            "charge": _parse(charge, dtype["charge"]),
            "particle_type": _parse(particle_type, dtype["particle_type"]),
            "sigma": _parse(sigma, dtype["sigma"]),
            "epsilon": _parse(epsilon, dtype["epsilon"]),
        }

        array_dict = {k: _parse(kwargs[k], dtype[k]) for k in dtype.names}
        try:
            df = pd.DataFrame(array_dict)
        except ValueError:
            df = pd.DataFrame(array_dict, index=[0])

        df = pd.concat((self.__self__.atomtypes, df), ignore_index=True)
        self.__self__.atomtypes = df[~df.duplicated("atom_type")].sort_values(
            ["atnum", "atom_type"], ignore_index=True,
        )

    @overload
    def nonbond_params(
        self,
        atom1: _ArrayLikeStr,
        atom2: _ArrayLikeStr,
        *,
        func: Literal[1],
        sigma: _ArrayLikeFloat = ...,
        epsilon: _ArrayLikeFloat = ...,
    ) -> None: ...

    @overload
    def nonbond_params(
        self,
        atom1: _ArrayLikeStr,
        atom2: _ArrayLikeStr,
        *,
        func: Literal[2],
        a: _ArrayLikeFloat = ...,
        b: _ArrayLikeFloat = ...,
        c: _ArrayLikeFloat = ...,
    ) -> None: ...

    def nonbond_params(
        self,
        atom1: _ArrayLikeStr,
        atom2: _ArrayLikeStr,
        *,
        func: Literal[1, 2],
        **kwargs: ArrayLike,
    ) -> None:
        """Add one or more atom types to the ``nonbond_params`` directive.

        Examples
        --------
        .. code-block:: python

            >>> import FOX

            >>> top: FOX.TOPContainer = ...
            >>> top.concatenate.nonbond_params(
            ...     ["C12", "C12"], ["C12", "H33"], func=1, epsilon=1.0, sigma=0.5
            ... )

        Parameters
        ----------
        atom1 : array-like
            One or more atom types for the first atom defining the bond
        atom2 : array-like
            One or more atom types for the second atom defining the bond
        func : {1, 2}
            The type of potential function for the non-bonded potential,
            1 representing Lennard-Jones and 2 Buckingham
        **kwargs : array-like
            Func-specific extra (optional) arguments:
            * 1: ``sigma`` and ``epsilon``
            * 2: ``a``, ``b`` and ``c``

        """
        dtype_dict = self.__self__.DF_DICT_DTYPES["nonbond_params"]
        try:
            dtype = dtype_dict[func]
        except KeyError:
            raise ValueError(f"Invalid func value: {func!r}") from None

        kwargs.update({"atom1": atom1, "atom2": atom2, "func": func})
        extra_keys = kwargs.keys() - set(dtype.names)
        if extra_keys:
            raise TypeError(
                f"nonbond_params() got {len(extra_keys)} unexpected keyword argument(s): "
                f"{', '.join(sorted(repr(i) for i in extra_keys))}"
            )

        array_dict = {k: _parse(kwargs.get(k, 0), dtype[k]) for k in dtype.names}
        try:
            df = pd.DataFrame(array_dict)
        except ValueError:
            df = pd.DataFrame(array_dict, index=[0])
        if (df_other := self.__self__.nonbond_params.get(func)) is not None:
            df = pd.concat((df_other, df), ignore_index=True)

        keys = ["atom1", "atom2"]
        self.__self__.nonbond_params[func] = df[~df.duplicated()].sort_values(
            keys, ignore_index=True
        )

    def atoms(
        self,
        atom_type: _ArrayLikeStr,
        molecule: _ArrayLikeStr,
        *,
        res_num: _ArrayLikeInt,
        res_name: _ArrayLikeStr,
        atom1: None | _ArrayLikeInt = None,
        atom_name: None | _ArrayLikeStr = None,
        charge_group: None | _ArrayLikeInt = None,
    ) -> None:
        """Add one or more atom types to the ``atoms`` directive.

        Examples
        --------
        .. code-block:: python

            >>> import FOX

            >>> top: FOX.TOPContainer = ...
            >>> top.concatenate.atoms(
            ...     molecule="mol1", res_num=5, res_name="OLA",
            ...     atom_type=["C12", "C12", "C13", "O38"],
            ... )

        Parameters
        ----------
        molecule : array-like
            One or more molecule names; must be present in the ``moleculetype`` directive
        res_num : array-like
            One or more residue numbers
        res_name : array-like
            One or more residue names
        atom_type : array-like
            One or more atom types
        atom1 : Array-like
            One or more atomic indices for the new atoms. Automatically inferred if unspecified.
        atom_name : array-like
            One or more atom names. Defaults to the same values as ``atom_type`` if unspecified.
        charge_group : array-like
            One or more charge groups. Defaults to the atomic index of unspecified.

        """
        dtype = self.__self__.DF_DTYPES["atoms"]
        kwargs = {
            "molecule": molecule,
            "atom1": 0 if atom1 is None else atom1,
            "atom_type": atom_type,
            "res_num": res_num,
            "res_name": res_name,
            "atom_name": atom_type if atom_name is None else atom_name,
            "charge_group": 0 if charge_group is None else charge_group,
            "charge": 0.0,
            "mass": 0.0,
        }

        array_dict = {k: _parse(kwargs[k], dtype[k]) for k in dtype.names}
        try:
            df = pd.DataFrame(array_dict)
        except ValueError:
            df = pd.DataFrame(array_dict, index=[0])

        df_other = self.__self__.atoms
        if atom1 is None:
            for mol in set(df_other["molecule"]):
                mask: pd.Series = df["molecule"] == mol
                mask_other: pd.Series = df_other["molecule"] == mol
                df_slice = df_other.loc[mask_other, "atom1"]
                start = 1 if df_slice.size == 0 else 1 + df_slice.max()
                df.loc[mask, "atom1"] = np.arange(start, start + mask.sum())

        if charge_group is None:
            df["charge_group"] = df["atom1"].copy()
        atomtypes_df = self.__self__.atomtypes.set_index("atom_type").loc[df["atom_type"], :]
        df[["charge", "mass"]] = atomtypes_df[["charge", "mass"]].values

        df = pd.concat((df_other, df), ignore_index=True)
        keys = ["molecule", "atom1"]
        self.__self__.atoms = df[~df.duplicated(keys)].sort_values(keys, ignore_index=True)

    def pairs(
        self,
        atom1: _ArrayLikeInt,
        atom2: _ArrayLikeInt,
        molecule: _ArrayLikeStr,
        *,
        func: Literal[1, 2],
    ) -> None:
        """Add one or more atom types to the ``pairs`` directive.

        Examples
        --------
        .. code-block:: python

            >>> import FOX

            >>> top: FOX.TOPContainer = ...
            >>> top.concatenate.pairs([1, 2, 3], [2, 3, 1], func=1)

        Parameters
        ----------
        molecule : array-like
            One or more molecule names; must be present in the ``moleculetype`` directive
        atom1 : array-like
            One or more atomic indices for the first atom defining the bond
        atom2 : array-like
            One or more atomic indices for the second atom defining the bond
        func : {1, 2}
            The type of potential function for the non-bonded potential,
            1 representing Lennard-Jones and 2 Buckingham

        """
        dtype = self.__self__.DF_DTYPES["pairs"]
        kwargs = {
            "molecule": molecule,
            "atom1": atom1,
            "atom2": atom2,
            "func": func,
        }

        array_dict = {k: _parse(kwargs[k], dtype[k]) for k in dtype.names}
        try:
            df = pd.DataFrame(array_dict)
        except ValueError:
            df = pd.DataFrame(array_dict, index=[0])

        df = pd.concat((self.__self__.pairs, df), ignore_index=True)
        keys = ["molecule", "atom1", "atom2"]
        self.__self__.pairs = df[~df.duplicated()].sort_values(keys, ignore_index=True)
