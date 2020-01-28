"""
FOX.io.read_prm
===============

A class for reading and generating .prm parameter files.

Index
-----
.. currentmodule:: FOX.io.read_prm
.. autosummary::
    PRMContainer

API
---
.. autoclass:: PRMContainer
    :members:
    :private-members:
    :special-members:

"""

import inspect
from types import MappingProxyType
from typing import Any, Iterator, Dict, Tuple, Set, Mapping, List, Union, Hashable
from itertools import chain
from collections import abc

import numpy as np
import pandas as pd
from assertionlib.dataclass import AbstractDataClass

from .file_container import AbstractFileContainer

__all__ = ['PRMContainer']


class PRMContainer(AbstractDataClass, AbstractFileContainer):
    """A container for managing prm files.

    Attributes
    ----------
    pd_printoptions : :class:`dict` [:class:`str`, :class:`object`], private
        A dictionary with Pandas print options.
        See `Options and settings <https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html>`_.

    """  # noqa

    #: A :class:`frozenset` with the names of private instance attributes.
    #: These attributes will be excluded whenever calling :meth:`PRMContainer.as_dict`.
    _PRIVATE_ATTR: Set[str] = frozenset({'_pd_printoptions'})

    #: A tuple of supported .psf headers.
    HEADERS: Tuple[str] = (
        'ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'NBFIX', 'HBOND', 'NONBONDED', 'IMPROPERS',
        'IMPROPER', 'END'
    )

    #: Define the columns for each DataFrame which hold its index
    INDEX: Mapping[str, List[int]] = MappingProxyType({
        'atoms': [2],
        'bonds': [0, 1],
        'angles': [0, 1, 2],
        'dihedrals': [0, 1, 2, 3],
        'nbfix': [0, 1],
        'nonbonded': [0],
        'impropers': [0, 1, 2, 3],
    })

    #: Placeholder values for DataFrame columns
    COLUMNS: Mapping[str, Tuple[Union[None, int, float], ...]] = MappingProxyType({
        'atoms': (None, -1, None, np.nan),
        'bonds': (None, None, np.nan, np.nan),
        'angles': (None, None, None, np.nan, np.nan, np.nan, np.nan),
        'dihedrals': (None, None, None, None, np.nan, -1, np.nan),
        'nbfix': (None, None, np.nan, np.nan),
        'nonbonded': (None, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan),
        'impropers': (None, None, None, None, np.nan, -1, np.nan)
    })

    def __init__(self, filename=None, atoms=None, bonds=None, angles=None, dihedrals=None,
                 impropers=None, nonbonded=None, nonbonded_header=None, nbfix=None,
                 hbond=None) -> None:
        """Initialize a :class:`PRMContainer` instance."""
        super().__init__()

        self.filename: str = filename
        self.atoms: pd.DataFrame = atoms
        self.bonds: pd.DataFrame = bonds
        self.angles: pd.DataFrame = angles
        self.dihedrals: pd.DataFrame = dihedrals
        self.impropers: pd.DataFrame = impropers
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
    def _read_iterate(cls, iterator):
        ret = {}
        value = None

        for i in iterator:
            i = i.rstrip('\n')
            if i.startswith('!') or i.startswith('*') or i.isspace() or not i:
                continue  # Ignore comment lines and empty lines

            key = i.split(maxsplit=1)[0]
            if key in cls.HEADERS:
                key = 'IMPROPERS' if key == 'IMPROPER' else key
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
        for key in self.HEADERS[:-2]:
            key_low = key.lower()
            df = getattr(self, key_low)
            if df is None:
                continue

            if key_low == 'hbond':
                write(f'\n{key} {df}\n')
                continue
            elif not isinstance(df, pd.DataFrame):
                continue

            iterator = range(df.shape[1] - 1)
            df_str = ' '.join('{:8}' for _ in iterator) + ' !{}\n'

            if key_low != 'nonbonded':
                write(f'\n{key}\n')
            else:
                header = '-\n'.join(i for i in self.nonbonded_header.split('-'))
                write(f'\n{key} {header}\n')
            for _, row_value in df.iterrows():
                write_str = df_str.format(*(('' if i is None else i) for i in row_value))
                write(write_str)

        write('\nEND\n')
