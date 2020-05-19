"""A class for reading and generating .prm parameter files.

Index
-----
.. currentmodule:: FOX.io.read_prm
.. autosummary::
    PRMContainer
    PRMContainer.read
    PRMContainer.write
    PRMContainer.overlay_mapping
    PRMContainer.overlay_cp2k_settings

API
---
.. autoclass:: PRMContainer
.. automethod:: PRMContainer.read
.. automethod:: PRMContainer.write
.. automethod:: PRMContainer.overlay_mapping
.. automethod:: PRMContainer.overlay_cp2k_settings

"""
import inspect
from types import MappingProxyType
from typing import (Any, Iterator, Dict, Tuple, FrozenSet, Mapping, List, Union, Iterable, Sequence,
                    Hashable, Optional, ClassVar, MutableSequence)
from itertools import chain, repeat
from contextlib import nullcontext
from collections import abc

import numpy as np
import pandas as pd

from scm.plams import Settings
from assertionlib.dataclass import AbstractDataClass

from .cp2k_to_prm import PRMMappingType, PostProcess
from .cp2k_to_prm import CP2K_TO_PRM as _CP2K_TO_PRM
from .file_container import AbstractFileContainer
from ..functions.cp2k_utils import parse_cp2k_value

__all__ = ['PRMContainer']

SeriesIdx = Mapping[str, float]  # e.g. a Pandas.Series with an Index
SeriesMultiIdx = Mapping[Tuple[str, ...], float]  # e.g. a Pandas.Series with a MultiIndex


class PRMContainer(AbstractDataClass, AbstractFileContainer):
    """A container for managing prm files.

    Attributes
    ----------
    pd_printoptions : :class:`dict` [:class:`str`, :class:`object`], private
        A dictionary with Pandas print options.
        See `Options and settings <https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html>`_.

    CP2K_TO_PRM : :class:`Mapping<collections.abc.Mapping>` [:class:`str`, :class:`PRMMapping<FOX.io.cp2k_to_prm.PRMMapping>`]
        A mapping providing tools for converting CP2K settings to .prm-compatible values.
        See :data:`CP2K_TO_PRM<FOX.io.cp2k_to_prm.CP2K_TO_PRM>`.

    """  # noqa

    #: A :class:`frozenset` with the names of private instance attributes.
    #: These attributes will be excluded whenever calling :meth:`PRMContainer.as_dict`.
    _PRIVATE_ATTR: ClassVar[FrozenSet[str]] = frozenset({'_pd_printoptions'})

    CP2K_TO_PRM: ClassVar[Mapping[str, PRMMappingType]] = _CP2K_TO_PRM

    #: A tuple of supported .psf headers.
    HEADERS: Tuple[str, ...] = (
        'ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'NBFIX', 'HBOND', 'NONBONDED', 'IMPROPER',
        'IMPROPERS', 'END'
    )

    #: Define the columns for each DataFrame which hold its index
    INDEX: Mapping[str, List[int]] = MappingProxyType({
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
    COLUMNS: Mapping[str, Tuple[Union[None, int, float], ...]] = MappingProxyType({
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

    def __init__(self, filename=None, atoms=None, bonds=None, angles=None, dihedrals=None,
                 improper=None, impropers=None, nonbonded=None, nonbonded_header=None, nbfix=None,
                 hbond=None) -> None:
        """Initialize a :class:`PRMContainer` instance."""
        super().__init__()

        self.filename: str = filename
        self.atoms: pd.DataFrame = atoms
        self.bonds: pd.DataFrame = bonds
        self.angles: pd.DataFrame = angles
        self.dihedrals: pd.DataFrame = dihedrals
        self.improper: pd.DataFrame = improper if improper is not None else impropers
        self.nonbonded_header: str = nonbonded_header
        self.nonbonded: pd.DataFrame = nonbonded
        self.nbfix: pd.DataFrame = nbfix
        self.hbond: str = hbond

        # Print options for Pandas DataFrames
        self.pd_printoptions: Dict[str, Any] = {'display.max_rows': 20}

    @property
    def pd_printoptions(self) -> Iterator[Union[Hashable, Any]]:
        return chain.from_iterable(self._pd_printoptions.items())

    @pd_printoptions.setter
    def pd_printoptions(self, value: Mapping[str, Any]) -> None:
        self._pd_printoptions = self._is_mapping(value)

    @staticmethod
    def _is_mapping(value: Any) -> dict:
        """Check if **value** is a :class:`dict` instance; raise a :exc:`TypeError` if not."""
        if not isinstance(value, abc.Mapping):
            caller_name: str = inspect.stack()[1][3]
            raise TypeError(f"The {repr(caller_name)} parameter expects an instance of 'dict'; "
                            f"observed type: '{value.__class__.__name__}''")
        return dict(value)

    @AbstractDataClass.inherit_annotations()
    def __repr__(self):
        with pd.option_context(*self.pd_printoptions):
            return super().__repr__()

    @AbstractDataClass.inherit_annotations()
    def __eq__(self, value):
        if type(self) is not type(value):
            return False

        try:
            for k, v1 in vars(self).items():
                if k in self._PRIVATE_ATTR:
                    continue

                v2 = getattr(value, k)
                if isinstance(v2, pd.DataFrame):
                    assert (v1 == v2).values.all()
                else:
                    assert v1 == v2
        except (AttributeError, AssertionError):
            return False
        else:
            return True

    # Ensure that a deepcopy is returned unless explictly specified

    @AbstractDataClass.inherit_annotations()
    def copy(self, deep=True): return super().copy(deep)

    @AbstractDataClass.inherit_annotations()
    def __copy__(self): return self.copy(deep=True)

    """########################### methods for reading .prm files. ##############################"""

    @classmethod
    @AbstractFileContainer.inherit_annotations()
    def read(cls, filename, encoding=None, **kwargs):
        return super().read(filename, encoding, **kwargs)

    @classmethod
    @AbstractFileContainer.inherit_annotations()
    def _read_iterate(cls, iterator):
        ret = {}
        value = None

        for i in iterator:
            i = i.rstrip('\n')
            if i.startswith('!') or i.startswith('*') or i.isspace() or not i:
                continue  # Ignore comment lines and empty lines

            key = i.split(maxsplit=1)[0]
            if key in cls.HEADERS:
                ret[key.lower()] = value = []
                ret[key.lower() + '_comment'] = value_comment = []
                if key in ('HBOND', 'NONBONDED'):
                    value.append(i.split()[1:])
                continue

            v, _, comment = i.partition('!')
            value.append(v.split())
            value_comment.append(comment.strip())

        cls._read_post_iterate(ret)
        return ret

    @classmethod
    def _read_post_iterate(cls, kwargs: dict) -> None:
        """Post process the dictionary produced by :meth:`PRMContainer._read_iterate`."""
        if 'end' in kwargs:
            del kwargs['end']
        if 'end_comment' in kwargs:
            del kwargs['end_comment']

        comment_dict = {}
        for k, v in kwargs.items():
            if k.endswith('_comment'):
                comment_dict[k] = v
            elif k == 'hbond':
                kwargs[k] = ' '.join(chain.from_iterable(v)).split('!')[0].rstrip()
            elif k == 'nonbonded':
                nonbonded_header = ' '.join(chain.from_iterable(v[0:2])).rstrip()
                kwargs[k] = df = pd.DataFrame(v[2:])
                cls._process_df(df, k)
            else:
                kwargs[k] = df = pd.DataFrame(v)
                cls._process_df(df, k)

        try:
            kwargs['nonbonded_header'] = nonbonded_header
        except NameError:
            pass

        for k, v in comment_dict.items():
            if k == 'nonbonded_comment':
                v = v[1:]
            del kwargs[k]
            if k == 'hbond_comment':
                continue
            kwargs[k.rstrip('_comment')]['comment'] = v

        for k, v in kwargs.items():
            if isinstance(v, pd.DataFrame) and not v.values.any():
                kwargs[k] = None

    @classmethod
    def _process_df(cls, df: pd.DataFrame, key: str) -> None:
        for i, default in enumerate(cls.COLUMNS[key]):
            if i not in df:
                df[i] = default
            else:
                default_type = str if default is None else type(default)
                df[i] = df[i].astype(default_type, copy=False)
        df['comment'] = None
        df.set_index(cls.INDEX[key], inplace=True)

    @AbstractFileContainer.inherit_annotations()
    def _read_postprocess(self, filename, encoding=None, **kwargs):
        if isinstance(filename, str):
            self.filename = filename

    """########################### methods for writing .prm files. ##############################"""

    @AbstractFileContainer.inherit_annotations()
    def write(self, filename=None, encoding=None, **kwargs):
        _filename = filename if filename is not None else self.filename
        if not _filename:
            raise TypeError("The 'filename' parameter is missing")
        super().write(_filename, encoding, **kwargs)

    @AbstractFileContainer.inherit_annotations()
    def _write_iterate(self, write, **kwargs) -> None:
        isnull = pd.isnull

        for key in self.HEADERS[:-2]:
            key_low = key.lower()
            df = getattr(self, key_low)

            if key_low == 'hbond' and df is not None:
                write(f'\n{key} {df}\n')
                continue
            elif not isinstance(df, pd.DataFrame):
                continue
            df = df.reset_index()

            iterator = range(df.shape[1] - 1)
            df_str = ' '.join('{:8}' for _ in iterator) + ' !{}\n'

            if key_low != 'nonbonded':
                write(f'\n{key}\n')
            else:
                header = '-\n'.join(i for i in self.nonbonded_header.split('-'))
                write(f'\n{key} {header}\n')
            for _, row_value in df.iterrows():
                write_str = df_str.format(*(('' if isnull(i) else i) for i in row_value))
                write(write_str)

        write('\nEND\n')

    """######################### Methods for updating the PRMContainer ##########################"""

    def overlay_mapping(self, prm_name: str,
                        param_df: Mapping[str, Union[SeriesIdx, SeriesMultiIdx]],
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

        param_df : :class:`pandas.DataFrame` or nested :class:`Mapping<collections.abc.Mapping>`
            A DataFrame or nested mapping with the to-be added parameters.
            The keys should be a subset of
            :attr:`PRMContainer.CP2K_TO_PRM[prm_name]["columns"]<PRMContainer.CP2K_TO_PRM>`.
            If the index/nested sub-keys consist of strings then they'll be split and turned into
            a :class:`pandas.MultiIndex`.
            Note that the resulting values are *not* sorted.

        units : :class:`Iterable<collections.abc.Iterable>` [:class:`str`], optional
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
        param_df = pd.DataFrame(param_df, copy=True)
        if not key_set.issuperset(param_df.columns):
            raise ValueError("The keys in `param_df` should be a subset of "
                             f"`PRMContainer.CP2K_TO_PRM[{prm_name!r}]['key']`")
        param_df.columns = [str2int[i] for i in param_df.columns]

        # Parse and validate the index
        if not isinstance(param_df.index, pd.MultiIndex):
            iterator = (i.split() for i in param_df.index)
            param_df.index = pd.MultiIndex.from_tuples(iterator)

        # Apply unit conversion and post-processing
        iterator = zip(param_df.items(), units, output_units, post_process)
        for (k, series), unit, output_unit, func in iterator:
            series_new = parse_cp2k_value(series, unit=output_unit, default_unit=unit)
            if func is not None:
                series_new = func(series_new)
            param_df[k] = series_new

        # Updated the DataFrame
        columns = param_df.columns
        for index, values in param_df.iterrows():
            df.loc[index, columns] = values

    def overlay_cp2k_settings(self, cp2k_settings: Mapping) -> None:
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
        cp2k_settings : :class:`Mapping<collections.abc.Mapping>`
            A Mapping with PLAMS-style CP2K settings.

        See Also
        --------
        PRMMapping : :class:`PRMMapping<FOX.io.cp2k_to_prm.PRMMapping>`
            A mapping providing tools for converting CP2K settings to .prm-compatible values.

        """
        if 'input' not in cp2k_settings:
            cp2k_settings = {'input': cp2k_settings}

        # If cp2k_settings is a Settings instance enable the supress_missing() context manager
        # In this manner normal KeyErrors will be raised, just like with dict
        if isinstance(cp2k_settings, Settings):
            context_manager = cp2k_settings.supress_missing
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

    def _overlay_cp2k_settings(self, cp2k_settings: Mapping,
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
