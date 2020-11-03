"""A :class:`~FOX.typed_mapping.TypedMapping` subclass converting CP2K settings to .prm-compatible values.

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

    A :class:`~collections.abc.Mapping` containing :class:`PRMMapping` instances.

"""  # noqa: E501

from types import MappingProxyType
from typing import Optional, Callable, Tuple, Mapping

from nanoutils import TypedDict

__all__ = ['PRMMapping', 'CP2K_TO_PRM']

PostProcess = Callable[[float], float]


class PRMMapping(TypedDict):
    """A :class:`TypedMapping<FOX.typed_mapping.TypedMapping>` providing tools for converting CP2K settings to .prm-compatible values.

    Attributes
    ----------
    name : :class:`str`
        The name of the :class:`~FOX.io.read_prm.PRMContainer` attribute.

    columns : :class:`tuple` [:class:`int`]
        The names relevant :class:`~FOX.io.read_prm.PRMContainer`
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

    post_process : :class:`tuple` [:data:`Callable[[float], float]<typing.Callable>`, optional]
        Callables for post-processing the value of interest.
        Set a particular callable to ``None`` to disable post-processing.

    """  # noqa: E501

    name: str
    columns: Tuple[int, ...]
    key_path: Tuple[str, ...]
    key: Tuple[str, ...]
    unit: Tuple[str, ...]
    default_unit: Tuple[Optional[str], ...]
    post_process: Tuple[Optional[PostProcess], ...]


def sigma_to_r2(sigma: float) -> float:
    r"""Convert :math:`\sigma` into :math:`\frac{1}{2} R`."""
    r = sigma * 2**(1/6)
    return r / 2


def return_zero(value: object) -> int:
    """Return :math:`0`."""
    return 0


CP2K_TO_PRM: Mapping[str, PRMMapping] = MappingProxyType({
    'nonbonded':
        PRMMapping(name='nbfix', columns=(2, 3),
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones'),  # noqa: E501
                   key=('epsilon', 'sigma'),
                   unit=('kcal/mol', 'angstrom'),
                   default_unit=('kcal/mol', 'kelvin'),
                   post_process=(None, sigma_to_r2)),

    'nonbonded14':
        PRMMapping(name='nbfix', columns=(4, 5),
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'nonbonded14', 'lennard-jones'),  # noqa: E501
                   key=('epsilon', 'sigma'),
                   unit=('kcal/mol', 'angstrom'),
                   default_unit=('kcal/mol', 'kelvin'),
                   post_process=(None, sigma_to_r2)),

    'bonds':
        PRMMapping(name='bonds', columns=(2, 3),
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'bond'),
                   key=('k', 'r0'),
                   unit=('kcal/mol/A**2', 'angstrom'),
                   default_unit=('internal_cp2k', 'bohr'),  # TODO: internal_cp2k ?????????
                   post_process=(None, None)),

    'angles':
        PRMMapping(name='angles', columns=(3, 4),
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'bend'),
                   key=('k', 'theta0'),
                   unit=('kcal/mol', 'degree'),
                   default_unit=('hartree', 'radian'),
                   post_process=(None, None)),

    'urrey-bradley':
        PRMMapping(name='angles', columns=(5, 6),
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'bend', 'ub'),
                   key=('k', 'r0'),
                   unit=('kcal/mol/A**2', 'angstrom'),
                   default_unit=('internal_cp2k', 'bohr'),  # TODO: internal_cp2k ?????????
                   post_process=(None, None)),

    'dihedrals':
        PRMMapping(name='dihedrals', columns=(4, 5, 6),
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'torsion'),
                   key=('k', 'm', 'phi0'),
                   unit=('kcal/mol', 'hartree', 'degree'),
                   default_unit=('hartree', 'hartree', 'radian'),
                   post_process=(None, None, None)),

    'improper':
        PRMMapping(name='improper', columns=(4, 5, 6),
                   key_path=('input', 'force_eval', 'mm', 'forcefield', 'improper'),
                   key=('k', 'k', 'phi0'),
                   unit=('kcal/mol', 'hartree', 'degree'),
                   default_unit=('hartree', 'hartree', 'radian'),
                   post_process=(None, return_zero, None)),
})
