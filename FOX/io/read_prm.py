"""A class for reading and generating .prm parameter files.

Index
-----
.. currentmodule:: FOX
.. autosummary::
    PRMContainer
    PRMContainer.read
    PRMContainer.write
    PRMContainer.overlay_mapping
    PRMContainer.overlay_cp2k_settings

API
---
.. autoclass:: PRMContainer
    :noindex:
    :members: atoms, bonds, angles, dihedrals, impropers, nonbonded, nbfix

.. automethod:: PRMContainer.read
.. automethod:: PRMContainer.write
.. automethod:: PRMContainer.overlay_mapping
.. automethod:: PRMContainer.overlay_cp2k_settings

"""
import copy
import textwrap
from types import MappingProxyType
from typing import (Any, Iterator, Dict, Tuple, Mapping, List, Union, Iterable, Sequence,
                    Optional, ClassVar, MutableSequence, Type, TypeVar, NamedTuple)
from itertools import chain, repeat
from contextlib import nullcontext

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from scm.plams import Settings
from nanoutils import AbstractFileContainer, set_docstring

from .cp2k_to_prm import CP2K_TO_PRM as _CP2K_TO_PRM, PRMMapping, PostProcess
from ..functions.cp2k_utils import parse_cp2k_value

__all__ = ['PRMContainer']

ST = TypeVar('ST', bound='PRMContainer')

# e.g. a Pandas.Series with an Index
SeriesIdx = Mapping[str, float]

# e.g. a Pandas.Series with a MultiIndex
SeriesMultiIdx = Mapping[Tuple[str, ...], float]


class _PRMAttrTup(NamedTuple):
    """A :class:`~collections.namedtuple` representing :class:`PRMContainer` attributes."""

    atoms: Optional[pd.DataFrame]
    bonds: Optional[pd.DataFrame]
    angles: Optional[pd.DataFrame]
    dihedrals: Optional[pd.DataFrame]
    impropers: Optional[pd.DataFrame]
    nbfix: Optional[pd.DataFrame]
    hbond: Optional[str]
    nonbonded_header: Optional[str]
    nonbonded: Optional[pd.DataFrame]


class PRMContainer(AbstractFileContainer):
    """A class for managing prm files.

    Examples
    --------
    .. code:: python

        >>> from FOX import PRMContainer

        >>> input_file = str(...)
        >>> output_file = str(...)

        >>> prm = PRMContainer.read(input_file)
        >>> prm.write(output_file)

    """

    # Attribute names should be in the same order as in __init__()
    __slots__ = _PRMAttrTup._fields + ('_pd_printoptions', '__weakref__')

    #: A dataframe holding atomic parameters.
    atoms: Optional[pd.DataFrame]

    #: A dataframe holding bond-related parameters.
    bonds: Optional[pd.DataFrame]

    #: A dataframe holding angle-related parameters.
    angles: Optional[pd.DataFrame]

    #: A dataframe holding proper dihedral-related parameters.
    dihedrals: Optional[pd.DataFrame]

    #: A dataframe holding improper diehdral-related parameters.
    impropers: Optional[pd.DataFrame]

    #: A dataframe holding non-bonded atomic parameters.
    nonbonded: Optional[pd.DataFrame]

    #: A string holding additional non-bonded related info.
    nonbonded_header: Optional[str]

    #: A dataframe holding non-bonded pair-wise atomic parameters.
    nbfix: Optional[pd.DataFrame]

    #: A string holding hydrogen bonding-related info.
    hbond: Optional[str]

    #: A mapping providing tools for converting CP2K settings to .prm-compatible values.
    #: See :data:`CP2K_TO_PRM<FOX.io.cp2k_to_prm.CP2K_TO_PRM>`.
    CP2K_TO_PRM: ClassVar[Mapping[str, PRMMapping]] = _CP2K_TO_PRM

    #: A tuple of supported .psf headers.
    _HEADERS: ClassVar[Tuple[str, ...]] = (
        'ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'NBFIX', 'HBOND', 'NONBONDED', 'IMPROPER',
        'IMPROPERS', 'END'
    )

    #: Define the columns for each DataFrame which hold its index
    _INDEX: ClassVar[Mapping[str, List[int]]] = MappingProxyType({
        'atoms': [2],
        'bonds': [0, 1],
        'angles': [0, 1, 2],
        'dihedrals': [0, 1, 2, 3],
        'nbfix': [0, 1],
        'nonbonded': [0],
        'improper': [0, 1, 2, 3],
        'impropers': [0, 1, 2, 3]
    })

    #: Placeholder values for DataFrame columns
    _COLUMNS: ClassVar[Mapping[str, Tuple[Any, ...]]] = MappingProxyType({
        'atoms': (None, -1, None, np.nan),
        'bonds': (None, None, np.nan, np.nan),
        'angles': (None, None, None, np.nan, np.nan, np.nan, np.nan),
        'dihedrals': (None, None, None, None, np.nan, -1, np.nan),
        'nbfix': (None, None, np.nan, np.nan, np.nan, np.nan),
        'nonbonded': (None, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        'improper': (None, None, None, None, np.nan, 0, np.nan),
        'impropers': (None, None, None, None, np.nan, 0, np.nan)
    })

    @property
    def improper(self) -> Optional[pd.DataFrame]:
        """Alias for :attr:`PRMContainer.impropers`."""
        return self.impropers

    @improper.setter
    def improper(self, value: Optional[pd.DataFrame]) -> None:
        self.impropers = value

    def __init__(self, atoms: Optional[pd.DataFrame] = None,
                 bonds: Optional[pd.DataFrame] = None,
                 angles: Optional[pd.DataFrame] = None,
                 dihedrals: Optional[pd.DataFrame] = None,
                 impropers: Optional[pd.DataFrame] = None,
                 nbfix: Optional[pd.DataFrame] = None,
                 hbond: Optional[str] = None,
                 nonbonded_header: Optional[str] = None,
                 nonbonded: Optional[pd.DataFrame] = None,
                 improper: Optional[pd.DataFrame] = None) -> None:
        """Initialize a :class:`PRMContainer` instance."""
        if impropers is not None and improper is not None:
            raise TypeError("'impropers' and 'improper' cannot be both specified")

        self.atoms = atoms
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals
        self.improper = improper if improper is not None else impropers
        self.nonbonded_header = nonbonded_header
        self.nonbonded = nonbonded
        self.nbfix = nbfix
        self.hbond = hbond

        # Print options for Pandas DataFrames
        self._pd_printoptions: Dict[str, Any] = {'display.max_rows': 20}

    @property
    def pd_printoptions(self) -> Iterator[Union[str, Any]]:
        """Return an iterator flattening :attr:`_pd_printoptions`."""
        return chain.from_iterable(self._pd_printoptions.items())

    def __repr__(self) -> str:
        """Implement :class:`str(self)<str>` and :func:`repr(self)<repr>`."""
        # Get all to-be printed attribute (names)
        cls = type(self)
        attr_names = cls.__slots__[:-2]

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

            return f'{cls.__name__}(\n{textwrap.indent(ret[:-2], 4 * " ")}\n)'

    def __eq__(self, value: object) -> bool:
        """Implement :meth:`self == value<object.__eq__>`."""
        if type(self) is not type(value):
            return False

        # Get all to-be printed attribute (names)
        cls = type(self)
        attr_names = cls.__slots__[:-2]

        # Compare the attributes
        ret = True
        str_or_none = {'nonbonded_header', 'hbond'}
        iterator = ((k, getattr(self, k), getattr(value, k)) for k in attr_names)
        for k, attr1, attr2 in iterator:
            if attr1 is attr2:
                continue
            elif k in str_or_none:
                ret &= attr1 == attr2
                continue

            try:
                assert_frame_equal(attr1, attr2)
            except AssertionError:
                return False
        return ret

    def __reduce__(self: ST) -> Tuple[Type[ST], _PRMAttrTup, Dict[str, Any]]:
        """Helper function for :mod:`pickle`."""
        cls = type(self)
        attr_names = cls.__slots__[:-2]
        attr_tup = _PRMAttrTup._make(getattr(self, k) for k in attr_names)
        return cls, attr_tup, self._pd_printoptions

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Helper function for :meth:`__reduce__`."""
        self._pd_printoptions = state

    def copy(self: ST, deep: bool = True) -> ST:
        """Create and return a copy of this instance.

        Parameters
        ----------
        deep : :class:`bool`
            If :data:`True`, return a deep copy.

        Returns
        -------
        :class:`FOX.PRMContainer`
            A new prmcontainer.

        """
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    """########################### methods for reading .prm files. ##############################"""

    @classmethod
    @set_docstring(AbstractFileContainer._read.__doc__)
    def _read(cls, file_obj, decoder):
        ret = {}
        special_header = {'hbond', 'nonbonded'}

        iterator = (decoder(i).rstrip('\n') for i in file_obj)
        for i in iterator:
            if i.startswith('!') or i.startswith('*') or i.isspace() or not i:
                continue  # Ignore comment lines and empty lines

            _key = i.split(maxsplit=1)[0]
            key = _key.lower()
            if _key in cls._HEADERS:
                ret[key] = value = []
                if key in special_header:
                    value.append(i.split()[1:])
                continue

            v, *_ = i.partition('!')
            value.append(v.split())

        cls._read_post_iterate(ret)
        return ret

    @classmethod
    def _read_post_iterate(cls, kwargs: dict) -> None:
        """Post process the dictionary produced by :meth:`PRMContainer._read_iterate`."""
        kwargs.pop('end', None)
        kwargs.pop('end_comment', None)

        nonbonded_header = None
        for k, v in kwargs.items():
            if k == 'hbond':
                kwargs[k] = ' '.join(chain.from_iterable(v)).split('!')[0].rstrip()
            elif k == 'nonbonded':
                nonbonded_header = ' '.join(chain.from_iterable(v[0:2])).rstrip()
                kwargs[k] = df = pd.DataFrame(v[2:])
                cls._process_df(df, k)
            else:
                kwargs[k] = df = pd.DataFrame(v)
                cls._process_df(df, k)
        kwargs['nonbonded_header'] = nonbonded_header

        for k, v in kwargs.items():
            if isinstance(v, pd.DataFrame):
                if not v.any().any():
                    kwargs[k] = None

    @classmethod
    def _process_df(cls, df: pd.DataFrame, key: str) -> None:
        """Fill in all columns, set their data type and assign an index to **df**."""
        for i, default in enumerate(cls._COLUMNS[key]):
            if i not in df:
                df[i] = default
            else:
                default_type = str if default is None else type(default)
                df[i] = df[i].astype(default_type, copy=False)
        df.set_index(cls._INDEX[key], inplace=True)

    """########################### methods for writing .prm files. ##############################"""

    @set_docstring(AbstractFileContainer._write.__doc__)
    def _write(self, file_obj, encoder) -> None:
        isnull = pd.isnull
        write = lambda n: file_obj.write(encoder(n))  # noqa: E731

        for key in self._HEADERS[:-2]:
            key_low = key.lower()
            df = getattr(self, key_low)

            if key_low == 'hbond' and df is not None:
                write(f'\n{key} {df}\n')
                continue
            elif not isinstance(df, pd.DataFrame):
                continue
            df = df.reset_index()  # Do NOT modify this inplace

            df_str = ('{:8} ' * df.shape[1])[:-1]

            if key_low != 'nonbonded':
                write(f'\n{key}\n')
            else:
                if self.nonbonded_header is not None:
                    header = '-\n'.join(i for i in self.nonbonded_header.split('-'))
                    write(f'\n{key} {header}\n')
            for _, row_value in df.iterrows():
                write_str = df_str.format(*(('' if isnull(i) else i) for i in row_value))
                write(f'{write_str.rstrip()}\n')
        write('\nEND\n')

    """######################### Methods for updating the PRMContainer ##########################"""

    def overlay_mapping(self, prm_name: str,
                        param: Mapping[str, Union[SeriesIdx, SeriesMultiIdx]],
                        units: Optional[Iterable[Optional[str]]] = None) -> None:
        """Update a set of parameters, **prm_name**, with those provided in **param_df**.

        Examples
        --------
        .. code:: python

            >>> from FOX import PRMContainer

            >>> prm = PRMContainer(...)

            >>> param_dict = {}
            >>> param_dict['epsilon'] = {'Cd Cd': ..., 'Cd Se': ..., 'Se Se': ...}  # epsilon
            >>> param_dict['sigma'] = {'Cd Cd': ..., 'Cd Se': ..., 'Se Se': ...}  # sigma

            >>> units = ('kcal/mol', 'angstrom')  # input units for epsilon and sigma

            >>> prm.overlay_mapping('nonbonded', param_dict, units=units)


        Parameters
        ----------
        prm_name : :class:`str`
            The name of the parameter of interest.
            See the keys of :attr:`PRMContainer.CP2K_TO_PRM` for accepted values.
        param : :class:`pandas.DataFrame` or nested :class:`~collections.abc.Mapping`
            A DataFrame or nested mapping with the to-be added parameters.
            The keys should be a subset of
            :attr:`PRMContainer.CP2K_TO_PRM[prm_name]["columns"]<PRMContainer.CP2K_TO_PRM>`.
            If the index/nested sub-keys consist of strings then they'll be split and turned into
            a :class:`pandas.MultiIndex`.
            Note that the resulting values are *not* sorted.
        units : :class:`Iterable[str] <collections.abc.Iterable>`, optional
            An iterable with the input units of each column in **param_df**.
            If ``None``, default to the defaults specified in
            :attr:`PRMContainer.CP2K_TO_PRM[prm_name]["unit"]<PRMContainer.CP2K_TO_PRM>`.

        """
        # Parse arguments
        if units is None:
            units = repeat(None)
        try:
            prm_map = self.CP2K_TO_PRM[prm_name]
        except KeyError as ex:
            raise ValueError(f"'prm_name is of invalid value ({prm_name!r}); "
                             f"accepted values: {tuple(self.CP2K_TO_PRM.keys())!r}") from ex

        # Extract parameter specific arguments
        name = prm_map['name']
        output_units = prm_map['unit']
        post_process = prm_map['post_process']
        key_set = set(prm_map['key'])
        str2int = {item: i for item, i in zip(prm_map['key'], prm_map['columns'])}

        # Ensure that the attribute in question is a DataFrame and not None
        df = getattr(self, name)
        if df is None:
            df = pd.DataFrame()
            setattr(self, name, df)
            self._process_df(df, name)

        # Parse and validate the columns
        param_df = pd.DataFrame(param, copy=True)
        if not key_set.issuperset(param_df.columns):
            raise ValueError("The keys in `param` should be a subset of "
                             f"`PRMContainer.CP2K_TO_PRM[{prm_name!r}]['key']`")
        param_df.columns = [str2int[i] for i in param_df.columns]

        # Parse and validate the index
        if not isinstance(param_df.index, pd.MultiIndex):
            iterator1 = (i.split() for i in param_df.index)
            param_df.index = pd.MultiIndex.from_tuples(iterator1)

        # Apply unit conversion and post-processing
        iterator2 = zip(param_df.items(), units, output_units, post_process)
        for (k, series), unit, output_unit, func in iterator2:
            series_new = parse_cp2k_value(series, unit=output_unit, default_unit=unit)
            if func is not None:
                series_new = func(series_new)
            param_df[k] = series_new

        # Updated the DataFrame
        columns = param_df.columns
        for index, values in param_df.iterrows():
            df.loc[index, columns] = values

    def overlay_cp2k_settings(self, cp2k_settings: Settings) -> None:
        """Extract forcefield information from PLAMS-style CP2K settings.

        Performs an inplace update of this instance.

        Examples
        --------
        Example input value for **cp2k_settings**.
        In the provided example the **cp2k_settings** are directly extracted from a CP2K .inp file.

        .. code:: python

            >>> import cp2kparser  # https://github.com/nlesc-nano/CP2K-Parser

            >>> filename = str(...)

            >>> cp2k_settings: dict = cp2kparser.read_input(filename)
            >>> print(cp2k_settings)
            {'force_eval': {'mm': {'forcefield': {'nonbonded': {'lennard-jones': [...]}}}}}


        Parameters
        ----------
        cp2k_settings : :class:`~collections.abc.Mapping`
            A Mapping with PLAMS-style CP2K settings.

        """
        if 'input' not in cp2k_settings:
            cp2k_settings = Settings({'input': cp2k_settings})

        # If cp2k_settings is a Settings instance enable the `suppress_missing` context manager
        # In this manner normal KeyErrors will be raised, just like with dict
        if isinstance(cp2k_settings, Settings):
            context_manager = cp2k_settings.suppress_missing
        else:
            context_manager = nullcontext

        with context_manager():
            for prm_map in self.CP2K_TO_PRM.values():
                name = prm_map['name']
                columns = list(prm_map['columns'])
                key_path = prm_map['key_path']
                key = prm_map['key']
                unit = prm_map['unit']
                default_unit = prm_map['default_unit']
                post_process = prm_map['post_process']

                self._overlay_cp2k_settings(cp2k_settings,
                                            name, columns, key_path, key, unit,
                                            default_unit, post_process)

    def _overlay_cp2k_settings(self, cp2k_settings: Settings,
                               name: str, columns: MutableSequence[int],
                               key_path: Sequence[str], key: Iterable[str],
                               unit: Iterable[str], default_unit: Iterable[Optional[str]],
                               post_process: Iterable[Optional[PostProcess]]) -> None:
        """Helper function for :meth:`PRMContainer.overlay_cp2k_settings`."""
        # Extract the appropiate dict or sequence of dicts
        try:
            prm_iter = Settings.get_nested(cp2k_settings, key_path)
        except KeyError:
            return
        else:
            prm_iter = (prm_iter,) if isinstance(prm_iter, Mapping) else prm_iter

        # Ensure that PRMContainter section is a DataFrame and not None
        df = getattr(self, name)
        if df is None:
            df = pd.DataFrame()
            setattr(self, name, df)
            self._process_df(df, name)

        # Extract, parse and write the values
        for i, prm_dict in enumerate(prm_iter):
            try:  # Extract the appropiate values
                index = prm_dict['atoms']
                value_gen = (prm_dict[k] for k in key)
            except KeyError as ex:
                raise KeyError(f"Failed to extract the {ex!r} key from "
                               f"{key_path[-1]!r} block {i}") from ex

            # Sanitize the values and convert them into appropiate units
            iterator = zip(value_gen, unit, default_unit)
            value_list = [parse_cp2k_value(*args) for args in iterator]

            # Post-process the values
            for i, (prm, func) in enumerate(zip(value_list, post_process)):
                if func is not None:
                    value_list[i] = func(prm)

            # Assign the values
            df.loc[index, columns] = value_list
