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
from typing import Any, Iterator, Dict, Tuple, Set, Mapping
from itertools import chain
from collections import abc

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
        'ATOMS', 'BONDS', 'ANGLES', 'DIHEDRALS', 'NBFIX', 'HBOND', 'NONBONDED', 'IMPROPER', 'END'
    )

    def __init__(self, filename=None, atoms=None, bonds=None, angles=None, dihedrals=None,
                 improper=None, nonbonded=None, nonbonded_header=None, nbfix=None,
                 hbond=None) -> None:
        """Initialize a :class:`PRMContainer` instance."""
        super().__init__()

        self.filename: str = filename
        self.atoms: pd.DataFrame = atoms
        self.bonds: pd.DataFrame = bonds
        self.angles: pd.DataFrame = angles
        self.dihedrals: pd.DataFrame = dihedrals
        self.improper: pd.DataFrame = improper
        self.nonbonded_header: str = nonbonded_header
        self.nonbonded: pd.DataFrame = nonbonded
        self.nbfix: pd.DataFrame = nbfix
        self.hbond: str = hbond

        # Print options for Pandas DataFrames
        self.pd_printoptions: Dict[str, Any] = {'display.max_rows': 20}

    @property
    def pd_printoptions(self) -> Iterator:
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
    def write(cls, filename=None, encoding=None, **kwargs):
        return super().write(filename, encoding, **kwargs)

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

    @staticmethod
    def _read_post_iterate(kwargs: dict) -> None:
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
                kwargs[k] = pd.DataFrame(v[2:])
            else:
                kwargs[k] = pd.DataFrame(v)

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
        for key in self.HEADERS[:-1]:
            key_low = key.lower()
            df = getattr(self, key_low)
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
