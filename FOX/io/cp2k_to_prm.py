"""
FOX.io.cp2k_to_prm
==================

A :class:`TypedMapping<FOX.typed_mapping.TypedMapping>` subclass converting CP2K settings to .prm-compatible values.

Index
-----
.. currentmodule:: FOX.io.cp2k_to_prm
.. autosummary::
    PRMMapping
    CP2K_TO_PRM

API
---
.. autoclass:: PRMMapping
.. autodata:: CP2K_TO_PRM
    :annotation: : MappingProxyType[str, PRMMapping]

    A :class:`Mapping<collections.abc.Mapping>` containing :class:`PRMMapping` instances.

    .. code:: python

        MappingProxyType({
            'nonbonded':
                PRMMapping(name='nbfix', columns=[2, 3],
                           key_path=('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones'),
                           key=('epsilon', 'sigma'),
                           unit=('kcal/mol', 'angstrom'),
                           default_unit=('kcal/mol', 'kelvin'),
                           post_process=(None, sigma_to_r2)),

            'nonbonded14':
                PRMMapping(name='nbfix', columns=[4, 5],
                           key_path=('input', 'force_eval', 'mm', 'forcefield', 'nonbonded14', 'lennard-jones'),
                           key=('epsilon', 'sigma'),
                           unit=('kcal/mol', 'angstrom'),
                           default_unit=('kcal/mol', 'kelvin'),
                           post_process=(None, sigma_to_r2)),

            'bonds':
                PRMMapping(name='bonds', columns=[2, 3],
                           key_path=('input', 'force_eval', 'mm', 'forcefield', 'bond'),
                           key=('k', 'r0'),
                           unit=('kcal/mol/A**2', 'angstrom'),
                           default_unit=('internal_cp2k', 'bohr'),  # TODO: internal_cp2k ?????????
                           post_process=(None, None)),

            'angles':
                PRMMapping(name='angles', columns=[3, 4],
                           key_path=('input', 'force_eval', 'mm', 'forcefield', 'bend'),
                           key=('k', 'theta0'),
                           unit=('kcal/mol', 'degree'),
                           default_unit=('hartree', 'radian'),
                           post_process=(None, None)),

            'urrey-bradley':
                PRMMapping(name='angles', columns=[5, 6],
                           key_path=('input', 'force_eval', 'mm', 'forcefield', 'bend', 'ub'),
                           key=('k', 'r0'),
                           unit=('kcal/mol/A**2', 'angstrom'),
                           default_unit=('internal_cp2k', 'bohr'),  # TODO: internal_cp2k ?????????
                           post_process=(None, None)),

            'dihedrals':
                PRMMapping(name='dihedrals', columns=[4, 5, 6],
                           key_path=('input', 'force_eval', 'mm', 'forcefield', 'torsion'),
                           key=('k', 'm', 'phi0'),
                           unit=('kcal/mol', 'hartree', 'degree'),
                           default_unit=('hartree', 'hartree', 'radian'),
                           post_process=(None, None, None)),

            'improper':
                PRMMapping(name='improper', columns=[4, 5, 6],
                           key_path=('input', 'force_eval', 'mm', 'forcefield', 'improper'),
                           key=('k', 'k', 'phi0'),
                           unit=('kcal/mol', 'hartree', 'degree'),
                           default_unit=('hartree', 'hartree', 'radian'),
                           post_process=(None, return_zero, None)),
        })

"""  # noqa

import sys
from types import MappingProxyType
from typing import (Optional, Iterable, Callable, FrozenSet, TypeVar, Tuple,
                    Union, Type, Any, ClassVar, Mapping)
from collections import abc

from ..typed_mapping import TypedMapping
from ..type_hints import TypedDict

__all__ = ['PRMMapping', 'CP2K_TO_PRM']

KV = TypeVar('KV')
NoneType = type(None)
PostProcess = Callable[[float], float]


class PRMMapping(TypedMapping):
    """A :class:`TypedMapping<FOX.typed_mapping.TypedMapping>` providing tools for converting CP2K settings to .prm-compatible values.

    Parameters
    ----------
    name : :class:`str`
        The name of the :class:`PRMContainer<FOX.io.read_prm.PRMContainer>` attribute.
        See :attr:`PRMMapping.name`.

    columns : :class:`int` or :class:`Iterable<collections.abc.Iterable>` [:class:`int`]
        The names relevant :class:`PRMContainer<FOX.io.read_prm.PRMContainer>`
        DataFrame columns.
        See :attr:`PRMMapping.columns`.

    key_path : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        The path of CP2K Settings keys leading to the property of interest.
        See :attr:`PRMMapping.key_path`.

    key : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        The key(s) within :attr:`PRMMapping.key_path` containg the actual properties of interest,
        *e.g.* ``"epsilon"`` and ``"sigma"``.
        See :attr:`PRMMapping.key`.

    unit : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        The desired output unit.
        See :attr:`PRMMapping.unit`.

    default_unit : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`, optional]
        The default unit as utilized by CP2K.
        See :attr:`PRMMapping.default_unit`.

    post_process : :class:`Callable<collections.abc.Callable>` or :class:`Iterable<collections.abc.Iterable>` [:class:`Callable<collections.abc.Callable>`]
        Callables for post-processing the value of interest.
        Set a particular callable to ``None`` to disable post-processing.
        See :attr:`PRMMapping.post_process`.

    Attributes
    ----------
    name : :class:`str`
        The name of the :class:`PRMContainer<FOX.io.read_prm.PRMContainer>` attribute.

    columns : :class:`tuple` [:class:`int`]
        The names relevant :class:`PRMContainer<FOX.io.read_prm.PRMContainer>`
        DataFrame columns.

    key_path : :class:`tuple` [:class:`str`]
        The path of CP2K Settings keys leading to the property of interest.

    key : :class:`tuple` [:class:`str`]
        The key(s) within :attr:`PRMMapping.key_path` containg the actual properties of interest,
        *e.g.* ``"epsilon"`` and ``"sigma"``.

    unit : :class:`tuple` [:class:`str`]
        The desired output unit.

    default_unit : :class:`tuple` [:class:`str`, optional]
        The default unit as utilized by CP2K.

    post_process : :class:`tuple` [:class:`Callable<collections.abc.Callable>`, optional]
        Callables for post-processing the value of interest.
        Set a particular callable to ``None`` to disable post-processing.

    """  # noqa

    _ATTR: ClassVar[FrozenSet[str]] = frozenset({
        'name', 'key', 'columns', 'key_path', 'unit', 'default_unit', 'post_process'
    })

    def __init__(self, name: str,
                 key: Union[str, Iterable[str]],
                 columns: Union[int, Iterable[int]],
                 key_path: Union[str, Iterable[str]],
                 unit: Union[str, Iterable[str]],
                 default_unit: Union[None, str, Iterable[Optional[str]]],
                 post_process: Union[None, PostProcess, Iterable[Optional[PostProcess]]]) -> None:
        """Initialize a :class:`PRMMapping` instance."""
        super().__init__()
        self.name = name
        self.key = self._to_tuple(key, str)
        self.columns = self._to_tuple(columns, int)
        self.key_path = self._to_tuple(key_path, str)
        self.unit = self._to_tuple(unit, str)
        self.default_unit = self._to_tuple(default_unit, (str, NoneType))
        self.post_process = self._to_tuple(post_process, (abc.Callable, NoneType))

    @staticmethod
    def _to_tuple(value: Union[KV, Iterable[KV]],
                  def_type: Union[Type[KV], Tuple[Type[KV], ...]]) -> Tuple[KV, ...]:
        """Convert **value** into a :class:`tuple`."""
        if isinstance(value, tuple):
            return value
        elif isinstance(value, def_type):
            return (value,)
        return tuple(value)


class _PRMMapping(TypedDict):
    name: str
    columns: Tuple[str, ...]
    key_path: Tuple[str, ...]
    key: Tuple[str, ...]
    unit: Tuple[str, ...]
    default_unit: Tuple[Optional[str], ...]
    post_process: Tuple[Optional[PostProcess], ...]


PRMMappingType = Union[PRMMapping, _PRMMapping]


def sigma_to_r2(sigma: float) -> float:
    r"""Convert :math:`\sigma` into :math:`\frac{1}{2} R`."""
    R = sigma * 2**(1/6)
    return R / 2


def return_zero(value: Any) -> int:
    """Return :math:`0`."""
    return 0


CP2K_TO_PRM: Mapping[str, PRMMappingType] = MappingProxyType({
    'nonbonded':
        PRMMapping(name='nbfix', columns=[2, 3],
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones'),  # noqa
                   key=('epsilon', 'sigma'),
                   unit=('kcal/mol', 'angstrom'),
                   default_unit=('kcal/mol', 'kelvin'),
                   post_process=(None, sigma_to_r2)),
    'nonbonded14':
        PRMMapping(name='nbfix', columns=[4, 5],
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'nonbonded14', 'lennard-jones'),  # noqa
                   key=('epsilon', 'sigma'),
                   unit=('kcal/mol', 'angstrom'),
                   default_unit=('kcal/mol', 'kelvin'),
                   post_process=(None, sigma_to_r2)),
    'bonds':
        PRMMapping(name='bonds', columns=[2, 3],
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'bond'),
                   key=('k', 'r0'),
                   unit=('kcal/mol/A**2', 'angstrom'),
                   default_unit=('internal_cp2k', 'bohr'),  # TODO: internal_cp2k ?????????
                   post_process=(None, None)),
    'angles':
        PRMMapping(name='angles', columns=[3, 4],
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'bend'),
                   key=('k', 'theta0'),
                   unit=('kcal/mol', 'degree'),
                   default_unit=('hartree', 'radian'),
                   post_process=(None, None)),
    'urrey-bradley':
        PRMMapping(name='angles', columns=[5, 6],
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'bend', 'ub'),
                   key=('k', 'r0'),
                   unit=('kcal/mol/A**2', 'angstrom'),
                   default_unit=('internal_cp2k', 'bohr'),  # TODO: internal_cp2k ?????????
                   post_process=(None, None)),
    'dihedrals':
        PRMMapping(name='dihedrals', columns=[4, 5, 6],
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'torsion'),
                   key=('k', 'm', 'phi0'),
                   unit=('kcal/mol', 'hartree', 'degree'),
                   default_unit=('hartree', 'hartree', 'radian'),
                   post_process=(None, None, None)),
    'improper':
        PRMMapping(name='improper', columns=[4, 5, 6],
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'improper'),
                   key=('k', 'k', 'phi0'),
                   unit=('kcal/mol', 'hartree', 'degree'),
                   default_unit=('hartree', 'hartree', 'radian'),
                   post_process=(None, return_zero, None)),
})
