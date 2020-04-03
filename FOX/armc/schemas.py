"""
FOX.armc.schemas
================

A module with template validation functions for the ARMC input.

Index
-----
.. currentmodule:: FOX.armc.schemas
.. autosummary::
{autosummary}

API
---
{autofunction}

"""

import os
from collections import abc
from typing import (overload, Any, SupportsInt, SupportsFloat, Type, Mapping, Collection, Sequence,
                    Callable, Union, Optional, Tuple, FrozenSet, Iterable, Dict, TypeVar)

import numpy as np
from schema import And, Or, Schema, Use, Optional as Optional_

from scm.plams import Settings as QmSettings
from qmflows import cp2k_mm
from qmflows.packages import Package

from .armc import ARMC
from .monte_carlo import MonteCarloABC
from .package_manager import PackageManager, PackageManagerABC
from .phi_updater import PhiUpdater, PhiUpdaterABC
from .param_mapping import ParamMapping, ParamMappingABC
from ..type_hints import Literal, SupportsArray, TypedDict, NDArray
from ..classes import MultiMolecule
from ..functions.utils import get_importable, _get_move_range

__all__ = [
    'validate_phi', 'validate_monte_carlo', 'validate_psf', 'validate_pes',
    'validate_job', 'validate_sub_job', 'validate_param', 'validate_main'
]

T = TypeVar('T')


@overload
def supports_float(value: SupportsFloat) -> Literal[True]: ...
@overload   # noqa: E302
def supports_float(value: Any) -> bool: ...
def supports_float(value):  # noqa: E302
    """Check if a float-like object has been passed (:data:`~typing.SupportsFloat`)."""
    try:
        value.__float__()
        return True
    except Exception:
        return False


@overload
def supports_int(value: SupportsInt) -> Literal[True]: ...
@overload   # noqa: E302
def supports_int(value: Any) -> bool: ...
def supports_int(value):  # noqa: E302
    """Check if a int-like object has been passed (:data:`~typing.SupportsInt`)."""
    # floats that can be exactly represented by an integer are also fine
    try:
        value.__int__()
        return float(value).is_integer()
    except Exception:
        return False


def phi_subclass(cls: type) -> bool:
    """Check if **cls** is a subclass of :class:`PhiUpdaterABC`."""
    return issubclass(cls, PhiUpdaterABC)


class Default:
    """A validation class akin to the likes of :class:`schemas.Use`.

    Upon executing :meth:`Default.validate` returns the stored :attr:`~Default.value`.
    If :attr:`~Default.call` is ``True`` and the value is a callable,
    then it is called before its return.

    Examples
    --------
    .. code:: python

        >>> from schema import Schema

        >>> schema1 = Schema(int, Default(True))
        >>> schema1.validate(1)
        True

        >>> schema2 = Schema(int, Default(dict))
        >>> schema2.validate(1)
        {}

        >>> schema3 = Schema(int, Default(dict, call=False))
        >>> schema3.validate(1)
        <class 'dict'>


    Attributes
    ----------
    value : :class:`~collections.abc.Callable` or :data:`~typing.Any`
        The to-be return value for when :meth:`Default.validate` is called.
        If :attr:`Default.call` is ``True`` then the value is called
        (if possible) before its return.
    call : :class:`bool`
        Whether to call :attr:`Default.value` before its return (if possible) or not.

    """

    value: Any
    call: bool

    def __init__(self, value: Union[T, Callable[[], T]], call: bool = True) -> None:
        self.value = value
        self.call = call

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.value!r}, call={self.call!r})'

    def validate(self, data: Any) -> Union[T, Callable[[], T]]:
        if self.call and callable(self.value):
            return self.value()
        else:
            return self.value


#: Schema for validating the ``"phi"`` block.
phi_schema = Schema({
    Optional_('type', default=lambda: PhiUpdater): Or(
        And(None, Default(PhiUpdater)),
        And(str, Use(lambda n: get_importable(n, validate=phi_subclass))),
        And(type, phi_subclass),
        error="'phi.type' expected a PhiUpdater class; observed value: {!r:100}"
    ),

    Optional_('phi', default=1.0): Or(
        And(None, Default(1.0)),
        And(supports_float, Use(float)),
        error="'phi.phi' expected a float; observed value: {!r:100}"
    ),

    Optional_('gamma', default=2.0): Or(
        And(None, Default(2.0)),
        And(supports_float, Use(float)),
        error="'phi.gamma' expected a float; observed value: {!r:100}"
    ),

    Optional_('a_target', default=0.25): Or(
        And(None, Default(0.25)),
        And(supports_float, lambda n: 0 < float(n) <= 1, Use(float)),
        error="'phi.a_target' expected a float in the (0, 1] interval; observed value: {!r:100}"
    ),

    Optional_('func', default=lambda: np.add): Or(
        And(None, Default(np.add, call=False)),
        And(str, Use(lambda n: get_importable(n, validate=callable))),
        abc.Callable,
        error="'phi.func' expected a Callable object; observed value: {!r:100}"
    ),

    Optional_('kwargs', default=dict): Or(
        And(None, Default(dict)),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        error="'phi.kwargs' expected a Mapping; observed value: {!r:100}"
    )
})


class PhiMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`phi_schema`."""

    type: Union[None, str, Type[PhiUpdaterABC]]
    phi: Optional[SupportsFloat]
    gamma: Optional[SupportsFloat]
    a_target: Optional[SupportsFloat]
    func: Union[None, str, Callable[[float, float], float]]
    kwargs: Optional[Mapping[str, Any]]


class PhiDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`phi_schema`."""

    type: Type[PhiUpdaterABC]
    phi: float
    gamma: float
    a_target: float
    func: Callable[[float, float], float]
    kwargs: Mapping[str, Any]


def mc_subclass(cls: type) -> bool:
    """Check if **cls** is a subclass of :class:`MonteCarloABC`."""
    return issubclass(cls, MonteCarloABC)


#: Schema for validating the ``"monte_carlo"`` block.
mc_schema = Schema({
    Optional_('type', default=lambda: ARMC): Or(
        And(None, Default(ARMC, call=False)),
        And(str, Use(lambda n: get_importable(n, validate=mc_subclass))),
        And(type, mc_subclass),
        error="'monte_carlo.type' expected an ARMC class; observed value: {!r:100}"
    ),

    Optional_('iter_len', default=50000): Or(
        And(None, Default(50000)),
        And(supports_int, lambda n: int(n) > 0, Use(int)),
        error="'monte_carlo.iter_len' expected a positive integer; observed value: {!r:100}"
    ),

    Optional_('sub_iter_len', default=100): Or(
        And(None, Default(100)),
        And(supports_int, lambda n: int(n) > 0, Use(int)),
        error="'monte_carlo.sub_iter_len' expected a positive integer; observed value: {!r:100}"
    ),

    Optional_('hdf5_file', default='armc.hdf5'): Or(
        And(None, Default('armc.hdf5')),
        str,
        os.PathLike,
        error="'monte_carlo.hdf5_file' expected a path-like object; observed value: {!r:100}"
    ),

    Optional_('logfile', default=lambda: 'armc.log'): Or(
        And(None, Default('armc.log')),
        str,
        os.PathLike,
        error="'monte_carlo.logfile' expected a path-like object; observed value: {!r:100}"
    ),

    Optional_('path', default=lambda: os.getcwd()): Or(
        And(None, Default(os.getcwd)),
        And(str, Use(os.path.abspath)),
        And(os.PathLike, Use(os.path.abspath)),
        error="'monte_carlo.path' expected a path-like object; observed value: {!r:100}"
    ),

    Optional_('folder', default='MM_MD_workdir'): Or(
        And(None, Default('MM_MD_workdir')),
        str,
        os.PathLike,
        error="'monte_carlo.folder' expected a path-like object; observed value: {!r:100}"
    ),

    Optional_('keep_files', default=True): Or(
        And(None, Default(True)),
        bool,
        error="'monte_carlo.keep_files' expected a boolean; observed value: {!r:100}"
    )
})


class MCMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`mc_schema`."""

    type: Union[None, str, Type[MonteCarloABC]]
    iter_len: Optional[SupportsInt]
    sub_iter_len:  Optional[SupportsInt]
    hdf5_file: Union[None, str, os.PathLike]
    logfile: Union[None, str, os.PathLike]
    path: Union[None, str, os.PathLike]
    folder: Union[None, str, os.PathLike]
    keep_files: Optional[bool]


class MCDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`mc_schema`."""

    type: Type[MonteCarloABC]
    iter_len: int
    sub_iter_len: int
    hdf5_file: Union[str, os.PathLike]
    logfile: Union[str, os.PathLike]
    path: Union[str, os.PathLike]
    folder: Union[str, os.PathLike]
    keep_files: bool


#: Schema for validating the ``"psf"`` block.
psf_schema = Schema({
    Optional_('str_file', default=None): Or(
        None,
        And(str, Use(lambda n: (n,))),
        And(os.PathLike, Use(lambda n: (n,))),
        And(abc.Sequence, lambda n: all(isinstance(i, (os.PathLike, str)) for i in n), Use(tuple)),
        error="'psf.str_file' expected None or one or more path-like objects; observed value: {!r:100}"
    ),

    Optional_('rtf_file', default=None): Or(
        None,
        And(str, Use(lambda n: (n,))),
        And(os.PathLike, Use(lambda n: (n,))),
        And(abc.Sequence, lambda n: all(isinstance(i, (os.PathLike, str)) for i in n), Use(tuple)),
        error="'psf.rtf_file' expected None or one or more path-like objects; observed value: {!r:100}"
    ),

    Optional_('psf_file', default=None): Or(
        None,
        And(str, Use(lambda n: (n,))),
        And(os.PathLike, Use(lambda n: (n,))),
        And(abc.Sequence, lambda n: all(isinstance(i, (os.PathLike, str)) for i in n), Use(tuple)),
        error="'psf.psf_file' expected None or one or more path-like objects; observed value: {!r:100}"
    ),

    Optional_('ligand_atoms', default=None): Or(
        None,
        And(str, Use(lambda n: frozenset({n}))),
        And(abc.Collection, lambda n: all(isinstance(i, str) for i in n), Use(frozenset)),
        error="'psf.ligand_atoms' expected None or a Collection of atoms; observed value: {!r:100}"
    ),
})


class PSFMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`psf_schema`."""

    str_file: Union[None, str, os.PathLike, Sequence[Union[str, os.PathLike]]]
    rtf_file: Union[None, str, os.PathLike, Sequence[Union[str, os.PathLike]]]
    psf_file: Union[None, str, os.PathLike, Sequence[Union[str, os.PathLike]]]
    ligand_atoms: Optional[Collection[str]]


class PSFDict(TypedDict):
    """A typed dict represneting the output of :data:`psf_schema`."""

    str_file: Optional[Tuple[Union[str, os.PathLike], ...]]
    rtf_file: Optional[Tuple[Union[str, os.PathLike], ...]]
    psf_file: Optional[Tuple[Union[str, os.PathLike], ...]]
    ligand_atoms: Optional[FrozenSet[str]]


#: Schema for validating the ``"pes"`` block.
pes_schema = Schema({
    'func': Or(
        And(str, Use(lambda n: get_importable(n, validate=callable))),
        abc.Callable,
        error="'pes.*.func' expected a callable object; observed value: {!r:100}"
    ),

    Optional_('kwargs', default=dict): Or(
        And(None, Default(dict)),
        And(abc.Mapping, lambda dct: all(isinstance(k, str) for k in dct.keys())),
        And(
            abc.Sequence,
            lambda n: all(isinstance(i, abc.Mapping) for i in n),
            lambda n: all(isinstance(k, str) for dct in n for k in dct.keys()),
            Use(tuple)
        ),
        error="'pes.*.kwargs' expected a Mapping or Sequence of Mappings; observed value: {!r:100}"
    )
})


class _PESMapping(TypedDict):
    func: Union[str, Callable]


class PESMapping(_PESMapping, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`pes_schema`."""

    kwargs: Union[None, Mapping[str, Any], Sequence[Mapping[str, Any]]]


class PESDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`pes_schema`."""

    func: Callable
    kwargs: Union[Mapping[str, Any], Tuple[Mapping[str, Any], ...]]


def pkg_subclass(cls: type) -> bool:
    """Check if **cls** is a subclass of :class:`PackageManagerABC`."""
    return issubclass(cls, PackageManagerABC)


#: Schema for validating the ``"job"`` block.
job_schema = Schema({
    Optional_('type', default=lambda: PackageManager): Or(
        And(None, Default(PackageManager, call=False)),
        And(str, Use(lambda n: get_importable(n, validate=pkg_subclass))),
        And(type, pkg_subclass),
        error="'job.func' expected a PackageManager class; observed value: {!r:100}"
    ),

    'molecule': Or(
        And(MultiMolecule, Use(lambda n: (n,))),
        And(str, Use(lambda n: (MultiMolecule.from_xyz(n),))),
        And(os.PathLike, Use(lambda n: (MultiMolecule.from_xyz(n),))),
        And(
            abc.Sequence,
            lambda n: all(isinstance(i, (str, os.PathLike, MultiMolecule)) for i in n),
            Use(lambda n: tuple(
                (i if isinstance(i, MultiMolecule) else MultiMolecule.from_xyz(i)) for i in n
            ))
        ),
        error="'job.molecule' expected one or more .xyz files; observed value: {!r:100}"
    )
})


class JobMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`job_schema`."""

    type: Union[None, str, Type[PackageManagerABC]]
    molecule: Union[MultiMolecule, str, os.PathLike, Sequence[Union[MultiMolecule, str, os.PathLike]]]  # noqa: E501


class JobDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`job_schema`."""

    type: Type[PackageManagerABC]
    molecule: Tuple[MultiMolecule, ...]


def qm_pkg_instance(obj: Any) -> bool:
    """Check if **cls** is a subclass of :class:`Package`."""
    return isinstance(obj, Package)


def mapping_instance(obj: Any) -> bool:
    """Check if **cls** is a subclass of :class:`Settings`."""
    return isinstance(obj, abc.Mapping)


#: Schema for validating sub blocks within the ``"pes"`` block.
sub_job_schema = Schema({
    Optional_('type', default=lambda: cp2k_mm): Or(
        And(None, Default(cp2k_mm, call=False)),
        And(str, Use(lambda n: get_importable(n, validate=qm_pkg_instance))),
        Package,
        error="'job.*.type' expected a Package instance; observed value: {!r}"
    ),

    Optional_('settings', default=QmSettings): Or(
        And(None, Default(QmSettings)),
        And(abc.Mapping, Use(QmSettings)),
        error="'job.*.settings' expected a Mapping; observed value: {!r}"
    ),

    Optional_('template', default=QmSettings): Or(
        And(None, Default(QmSettings)),
        And(str, Use(lambda n: QmSettings(get_importable(n, validate=mapping_instance)))),
        And(abc.Mapping, Use(QmSettings)),
        error="'job.*.template' expected a Mapping; observed value: {!r}"
    )
})


class SubJobMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`sub_job_schema`."""

    type: Union[None, str, Type[Package]]
    settings: Optional[Mapping]
    template: Union[None, str, Mapping]


class SubJobDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`sub_job_schema`."""

    type: Type[Package]
    settings: QmSettings
    template: QmSettings


MOVE_DEFAULT = np.array([
    0.900, 0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935, 0.940,
    0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985,
    0.990, 0.995, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030, 1.035,
    1.040, 1.045, 1.050, 1.055, 1.060, 1.065, 1.070, 1.075, 1.080,
    1.085, 1.090, 1.095, 1.100
], dtype=float)
MOVE_DEFAULT.setflags(write=False)


def prm_subclass(cls: type) -> bool:
    """Check if **cls** is a subclass of :class:`ParamMappingABC`."""
    return issubclass(cls, ParamMappingABC)


#: Schema for validating the ``"param"`` block.
param_schema = Schema({
    Optional_('type', default=lambda: ParamMapping): Or(
        And(None, Default(ParamMapping, call=False)),
        And(str, Use(lambda n: get_importable(n, validate=prm_subclass))),
        And(type, prm_subclass),
        error="'param.type' expected a ParamMapping class; observed value: {!r}"
    ),

    Optional_('func', default=lambda: np.multiply): Or(
        And(None, Default(np.multiply, call=False)),
        And(str, Use(lambda n: get_importable(n, callable))),
        abc.Callable,
        error="'param.func' expected a Callable object; observed value: {!r}"
    ),

    Optional_('kwargs', default=dict): Or(
        And(None, Default(dict)),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        error="'param.kwargs' expected a Mapping; observed value: {!r}"
    ),

    Optional_('move_range', default=lambda: MOVE_DEFAULT.copy()): Or(
        And(None, Default(MOVE_DEFAULT.copy)),
        And(abc.Sequence, Use(lambda n: np.asarray(n, dtype=float))),
        And(SupportsArray, Use(lambda n: np.asarray(n, dtype=float))),
        And(abc.Iterable, Use(lambda n: np.fromiter(n, dtype=float))),
        And(abc.Mapping, lambda n: {'start', 'stop', 'step'}.issuperset(n.keys()),
            Use(lambda n: _get_move_range(**n))),
        error="'param.move_range' expected a Mapping or array-like object; observed value: {!r}"
    )

})


class MoveRange(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the ``"move_range"`` block in :data:`param_schema`."""  # noqa: E501

    start: float
    stop: float
    step: float


class ParamMapping_(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`param_schema`."""

    type: Union[None, str, Type[ParamMappingABC]]
    func: Union[None, str, Callable[[float, float], float]]
    kwargs: Optional[Mapping[str, Any]]
    move_range: Union[None, Iterable, SupportsArray, MoveRange]


class ParamDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`param_schema`."""

    type: Type[ParamMappingABC]
    func: Callable[[float, float], float]
    kwargs: Mapping[str, Any]
    move_range: NDArray[float]


main_schema = Schema({
    'param': Or(
        And(abc.MutableMapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys()), Use(dict))
    ),

    'pes': Or(
        And(abc.MutableMapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys()), Use(dict))
    ),

    'job': Or(
        And(abc.MutableMapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys()), Use(dict))
    ),

    Optional_('monte_carlo', default=dict): Or(
        And(None, Default(dict)),
        And(abc.MutableMapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys()), Use(dict))
    ),

    Optional_('phi', default=dict): Or(
        And(None, Default(dict)),
        And(abc.MutableMapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys()), Use(dict))
    ),

    Optional_('psf', default=dict): Or(
        And(None, Default(dict)),
        And(abc.MutableMapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys()), Use(dict))
    )
})


class _MainMapping(TypedDict):
    param: ParamMapping_
    pes: Mapping[str, PESMapping]
    job: JobMapping


class MainMapping(_MainMapping, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`main_schema`."""

    monte_carlo: Optional[MCMapping]
    phi: Optional[PhiMapping]
    psf: Optional[PSFMapping]


class MainDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`main_schema`."""

    param: ParamMapping_
    pes: Dict[str, PESMapping]
    job: JobMapping
    monte_carlo: MCMapping
    phi: PhiMapping
    psf: PSFMapping


def validate_main(mapping: MainMapping) -> MainDict:
    """Validate the all super-keys."""
    return main_schema.validate(mapping)


def validate_phi(mapping: PhiMapping) -> PhiDict:
    """Validate the ``"phi"`` block."""
    return phi_schema.validate(mapping)


def validate_monte_carlo(mapping: MCMapping) -> MCDict:
    """Validate the ``"monte_carlo"`` block."""
    return mc_schema.validate(mapping)


def validate_psf(mapping: PSFMapping) -> PSFDict:
    """Validate the ``"psf"`` block."""
    return psf_schema.validate(mapping)


def validate_pes(mapping: PESMapping) -> PESDict:
    """Validate the ``"pes"`` block."""
    return pes_schema.validate(mapping)


def validate_job(mapping: JobMapping) -> JobDict:
    """Validate the ``"job"`` block."""
    return job_schema.validate(mapping)


def validate_sub_job(mapping: SubJobMapping) -> SubJobDict:
    """Validate sub-blocks within the ``"job"`` block."""
    return sub_job_schema.validate(mapping)


def validate_param(mapping: ParamMapping_) -> ParamDict:
    """Validate the ``"param"`` block."""
    return param_schema.validate(mapping)


__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autofunction='\n'.join(f'.. autofunction:: {i}' for i in __all__)
)
