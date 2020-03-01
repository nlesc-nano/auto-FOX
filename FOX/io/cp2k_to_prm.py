from typing import Optional, Iterable, Callable, FrozenSet, TypeVar, Tuple, Union, Type, Any
from collections import abc

from ..frozen_mapping import FrozenMapping

__all__ = ['PRMMapping', 'CP2K_TO_PRM']

KV = TypeVar('KV')
NoneType = type(None)
PostProcess = Callable[[float], float]


class PRMMapping(FrozenMapping):
    """A mapping providing tools for converting CP2K settings to .prm-compatible values.

    Attributes
    ----------
    name : :class:`str`
        The name of the :class:`PRMContainer<FOX.io.read_prm.PRMContainer>` attribute.

    columns : :class:`int` or :class:`Iterable<collections.abc.Iterable>` [:class:`int`]
        The names relevant :class:`PRMContainer<FOX.io.read_prm.PRMContainer>`
        DataFrame columns.

    key_path : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        The path of CP2K Settings keys leading to the property of interest.

    key : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        The key(s) within :attr:`PRMMapping.key_path` containg the actual properties of interest,
        *e.g.* ``"epsilon"`` and ``"sigma"``.

    unit : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        The desired output unit.

    default_unit : :class:`str` or :class:`Iterable<collections.abc.Iterable>` [:class:`str`]
        The default unit as utilized by CP2K.

    post_process : :class:`Callable<collections.abc.Callable>` or :class:`Iterable<collections.abc.Iterable>` [:class:`Callable<collections.abc.Callable>`]
        Callables for post-processing the value of interest.
        Set a particular callable to ``None`` to disable post-processing.

    """  # noqa

    def __init__(self, name: str,
                 columns: Union[int, Iterable[int]],
                 key_path: Union[str, Iterable[str]],
                 key: Union[str, Iterable[str]],
                 unit: Union[str, Iterable[str]],
                 default_unit: Union[None, str, Iterable[Optional[str]]],
                 post_process: Union[None, PostProcess, Iterable[Optional[PostProcess]]]) -> None:
        super().__init__()
        self.name = name
        self.columns = self._to_tuple(columns, int)
        self.key_path = self._to_tuple(key_path, str)
        self.key = self._to_tuple(key, str)
        self.unit = self._to_tuple(unit, str)
        self.default_unit = self._to_tuple(default_unit, (str, NoneType))
        self.post_process = self._to_tuple(post_process, (abc.Callable, NoneType))

    @staticmethod
    def _to_tuple(value: Union[KV, Iterable[KV]],
                  def_type: Union[Type[KV], Tuple[Type[KV], ...]]) -> Tuple[KV, ...]:
        if isinstance(value, tuple):
            return value
        elif isinstance(value, def_type):
            return (value,)
        return tuple(value)


def sigma_to_r2(sigma: float) -> float:
    r"""Convert :math:`\sigma` into :math:`|frac{1}{2} R`."""
    R = sigma * 2**(1/6)
    return R / 2


def return_zero(value: Any) -> int:
    """Return :math:`0`."""
    return 0


#: A :class:`frozenset`
CP2K_TO_PRM: FrozenSet[PRMMapping] = frozenset({
    PRMMapping(name='nbfix', columns=[2, 3],
               key_path=('input', 'force_eval', 'mm', 'forcefield', 'nonbonded', 'lennard-jones'),
               key=('epsilon', 'sigma'),
               unit=('kcal/mol', 'angstrom'),
               default_unit=('kcal/mol', 'kelvin'),
               post_process=(None, sigma_to_r2)),

    PRMMapping(name='nbfix', columns=[4, 5],
               key_path=('input', 'force_eval', 'mm', 'forcefield', 'nonbonded14', 'lennard-jones'),
               key=('epsilon', 'sigma'),
               unit=('kcal/mol', 'angstrom'),
               default_unit=('kcal/mol', 'kelvin'),
               post_process=(None, sigma_to_r2)),

    PRMMapping(name='bonds', columns=[2, 3],
               key_path=('input', 'force_eval', 'mm', 'forcefield', 'bond'),
               key=('k', 'r0'),
               unit=('kcal/mol/A**2', 'angstrom'),
               default_unit=('internal_cp2k', 'bohr'),  # TODO: internal_cp2k ?????????
               post_process=(None, None)),

    PRMMapping(name='angles', columns=[3, 4],
               key_path=('input', 'force_eval', 'mm', 'forcefield', 'bend'),
               key=('k', 'theta0'),
               unit=('kcal/mol', 'degree'),
               default_unit=('hartree', 'radian'),
               post_process=(None, None)),

    PRMMapping(name='angles', columns=[5, 6],
               key_path=('input', 'force_eval', 'mm', 'forcefield', 'bend', 'ub'),
               key=('k', 'r0'),
               unit=('kcal/mol/A**2', 'angstrom'),
               default_unit=('internal_cp2k', 'bohr'),  # TODO: internal_cp2k ?????????
               post_process=(None, None)),

    PRMMapping(name='dihedrals', columns=[4, 5, 6],
               key_path=('input', 'force_eval', 'mm', 'forcefield', 'torsion'),
               key=('k', 'm', 'phi0'),
               unit=('kcal/mol', 'hartree', 'degree'),
               default_unit=('hartree', 'hartree', 'radian'),
               post_process=(None, None, None)),

    PRMMapping(name='improper', columns=[4, 5, 6],
               key_path=('input', 'force_eval', 'mm', 'forcefield', 'improper'),
               key=('k', 'k', 'phi0'),
               unit=('kcal/mol', 'hartree', 'degree'),
               default_unit=('hartree', 'hartree', 'radian'),
               post_process=(None, return_zero, None)),
})
