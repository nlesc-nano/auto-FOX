"""A module with template validation functions for the ARMC input.

Index
-----
.. currentmodule:: FOX.armc.schemas
.. autosummary::
{autosummary}

API
---
{autofunction}

"""

from __future__ import annotations

import os
from functools import partial
from collections import abc
from typing import (
    Any, SupportsInt, Type, Mapping, Collection, Sequence, SupportsFloat, Generator,
    Callable, Union, Optional, Tuple, FrozenSet, Iterable, Dict, TypeVar, List,
    cast, TYPE_CHECKING,
)

import numpy as np
from schema import And, Or, Schema, Use, Optional as Optional_

from qmflows import cp2k_mm, Settings as QmSettings
from qmflows.packages import Package
from nanoutils import (
    Default, Formatter, supports_int, supports_float, isinstance_factory,
    issubclass_factory, import_factory, as_nd_array, TypedDict, Literal,
)

from .armc import ARMC
from .monte_carlo import MonteCarloABC
from .package_manager import PackageManager, PackageManagerABC
from .phi_updater import PhiUpdater, PhiUpdaterABC
from .param_mapping import ParamMapping, ParamMappingABC
from .err_funcs import default_error_func
from ..type_hints import ArrayLike, Scalar, ArrayLikeOrScalar
from ..classes import MultiMolecule
from ..utils import get_move_range
from ..io.cp2k import lattice_from_cell
from ..io.read_xyz import read_multi_xyz

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    'validate_phi', 'validate_monte_carlo', 'validate_psf', 'validate_pes',
    'validate_job', 'validate_sub_job', 'validate_param', 'validate_main',
    'validation_schema',
]

T = TypeVar('T')

EPILOG = '.\n\n{name}: {type} = {value!r:.100}'

# Create a number of callables which take a single parameter is argument
# This ensures they can be used with :class:`schema.Use`.

phi_subclass = issubclass_factory(PhiUpdaterABC)
mc_subclass = issubclass_factory(MonteCarloABC)
pkg_subclass = issubclass_factory(PackageManagerABC)
prm_subclass = issubclass_factory(ParamMappingABC)
qm_pkg_instance = isinstance_factory(Package)
mapping_instance = isinstance_factory(abc.Mapping)

import_phi = import_factory(validate=phi_subclass)
import_mc = import_factory(validate=mc_subclass)
import_pkg = import_factory(validate=pkg_subclass)
import_prm = import_factory(validate=prm_subclass)
import_qm_pkg = import_factory(validate=qm_pkg_instance)
import_mapping = import_factory(validate=mapping_instance)

import_func = import_factory(validate=callable)

array: Callable[[Any], np.ndarray]
array = partial(as_nd_array, ndmin=1, dtype=float, copy=True)

#: The default value for ``phi.phi```.
PHI = np.array([1.0])
PHI.setflags(write=False)

#: The default value for ``phi.gamma```.
GAMMA = np.array([2.0])
GAMMA.setflags(write=False)

#: The default value for ``phi.a_target```.
A_TARGET = np.array([0.25])
A_TARGET.setflags(write=False)


def abspath(file: str | bytes | os.PathLike[Any]) -> str:
    """Return the absolute path of **file**."""
    file_str = os.fsdecode(file)
    return os.path.abspath(file_str)


CellFunc = Callable[[Union[str, bytes, "os.PathLike[Any]"]], "NDArray[np.float64]"]
EXT_DICT: Dict[Literal["cell", "xyz"], CellFunc] = {
    "cell": lattice_from_cell,
    "xyz": lambda n: read_multi_xyz(n)[0],
}


def parse_lattice(file: str | bytes | os.PathLike[Any]) -> NDArray[np.float64]:
    file_str = os.fsdecode(file)
    _, ext = os.path.splitext(file_str)
    ext = cast(Literal['cell'], ext.strip("."))
    try:
        func = EXT_DICT[ext]
    except KeyError:
        raise ValueError(f"Unsupported file extensions: {ext!r}") from None
    return func(file_str)


#: Schema for validating the ``"phi"`` block.
phi_schema = Schema({
    Optional_('type', default=lambda: PhiUpdater): Or(
        And(None, Default(PhiUpdater)),
        And(str, Use(import_phi)),
        And(type, phi_subclass),
        error=Formatter(f"'phi.type' expected a PhiUpdater class; observed value{EPILOG}")
    ),

    Optional_('phi', default=PHI.copy): Or(
        And(None, Default(PHI.copy)),
        And(Use(array)),
        error=Formatter(f"'phi.phi' expected a float{EPILOG}")
    ),

    Optional_('gamma', default=GAMMA.copy): Or(
        And(None, Default(GAMMA.copy)),
        And(Use(array)),
        error=Formatter(f"'phi.gamma' expected a float{EPILOG}")
    ),

    Optional_('a_target', default=A_TARGET.copy): Or(
        And(None, Default(A_TARGET.copy)),
        And(lambda n: (0 < array(n)).all() & (1 >= array(n)).all(), Use(array)),
        error=Formatter("'phi.a_target' expected a float or sequence of floats "
                        f"in the (0, 1] interval{EPILOG}")
    ),

    Optional_('func', default=lambda: np.add): Or(
        And(None, Default(np.add, call=False)),
        And(str, Use(import_func)),
        abc.Callable,
        error=Formatter(f"'phi.func' expected a Callable object{EPILOG}")
    ),

    Optional_('kwargs', default=dict): Or(
        And(None, Default(dict)),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        error=Formatter(f"'phi.kwargs' expected a Mapping{EPILOG}")
    )
}, name='phi_schema', description='Schema for validating the "phi" block.')


PhiFunc = Callable[[np.ndarray, ArrayLikeOrScalar], np.ndarray]


class PhiMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`phi_schema`."""

    type: Union[None, str, Type[PhiUpdaterABC]]
    phi: Union[None, ArrayLike, Iterable[Scalar]]
    gamma: Union[None, ArrayLike, Iterable[Scalar]]
    a_target: Union[None, ArrayLike, Iterable[Scalar]]
    func: Union[None, str, PhiFunc]
    kwargs: Optional[Mapping[str, Any]]


class PhiDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`phi_schema`."""

    type: Type[PhiUpdaterABC]
    phi: np.ndarray
    gamma: np.ndarray
    a_target: np.ndarray
    func: PhiFunc
    kwargs: Mapping[str, Any]


#: Schema for validating the ``"monte_carlo"`` block.
mc_schema = Schema({
    Optional_('type', default=lambda: ARMC): Or(
        And(None, Default(ARMC, call=False)),
        And(str, Use(import_mc)),
        And(type, mc_subclass),
        error=Formatter(f"'monte_carlo.type' expected an ARMC class{EPILOG}")
    ),

    Optional_('iter_len', default=50000): Or(
        And(None, Default(50000)),
        And(supports_int, lambda n: int(n) > 0, Use(int)),
        error=Formatter(f"'monte_carlo.iter_len' expected a positive integer{EPILOG}")
    ),

    Optional_('sub_iter_len', default=100): Or(
        And(None, Default(100)),
        And(supports_int, lambda n: int(n) > 0, Use(int)),
        error=Formatter(f"'monte_carlo.sub_iter_len' expected a positive integer{EPILOG}")
    ),

    Optional_('hdf5_file', default='armc.hdf5'): Or(
        And(None, Default('armc.hdf5')),
        And(Or(str, bytes, os.PathLike), Use(os.fsdecode)),
        error=Formatter(f"'monte_carlo.hdf5_file' expected a path-like object{EPILOG}")
    ),

    Optional_('logfile', default=lambda: 'armc.log'): Or(
        And(None, Default('armc.log')),
        And(Or(str, bytes, os.PathLike), Use(os.fsdecode)),
        error=Formatter(f"'monte_carlo.logfile' expected a path-like object{EPILOG}")
    ),

    Optional_('path', default=lambda: os.getcwd()): Or(
        And(None, Default(os.getcwd)),
        And(Or(str, bytes, os.PathLike), Use(abspath)),
        error=Formatter(f"'monte_carlo.path' expected a path-like object{EPILOG}")
    ),

    Optional_('folder', default='MM_MD_workdir'): Or(
        And(None, Default('MM_MD_workdir')),
        And(Or(str, bytes, os.PathLike), Use(os.fsdecode)),
        error=Formatter(f"'monte_carlo.folder' expected a path-like object{EPILOG}")
    ),

    Optional_('keep_files', default=True): Or(
        And(None, Default(True)),
        bool,
        error=Formatter(f"'monte_carlo.keep_files' expected a boolean{EPILOG}")
    )
}, name='mc_schema', description='Schema for validating the "monte_carlo" block.')


class MCMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`mc_schema`."""

    type: Union[None, str, Type[MonteCarloABC]]
    iter_len: Union[None, str, bytes, SupportsInt]
    sub_iter_len: Union[None, str, bytes, SupportsInt]
    hdf5_file: Union[None, str, bytes, os.PathLike]
    logfile: Union[None, str, bytes, os.PathLike]
    path: Union[None, str, bytes, os.PathLike]
    folder: Union[None, str, bytes, os.PathLike]
    keep_files: Optional[bool]


class MCDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`mc_schema`."""

    type: Type[MonteCarloABC]
    iter_len: int
    sub_iter_len: int
    hdf5_file: str
    logfile: str
    path: str
    folder: str
    keep_files: bool


#: Schema for validating the ``"psf"`` block.
psf_schema = Schema({
    Optional_('str_file', default=None): Or(
        None,
        And(Or(str, bytes, os.PathLike), Use(lambda n: (abspath(n),))),
        And(abc.Sequence, lambda n: all(isinstance(i, (os.PathLike, bytes, str)) for i in n),  # type: ignore[misc] # noqa: E501
            Use(lambda n: tuple(abspath(i) for i in n))),
        error=Formatter(f"'psf.str_file' expected None or one or more path-like objects{EPILOG}")
    ),

    Optional_('rtf_file', default=None): Or(
        None,
        And(Or(str, bytes, os.PathLike), Use(lambda n: (abspath(n),))),
        And(abc.Sequence, lambda n: all(isinstance(i, (os.PathLike, bytes, str)) for i in n),  # type: ignore[misc] # noqa: E501
            Use(lambda n: tuple(abspath(i) for i in n))),
        error=Formatter(f"'psf.rtf_file' expected None or one or more path-like objects{EPILOG}")
    ),

    Optional_('psf_file', default=None): Or(
        None,
        And(Or(str, bytes, os.PathLike), Use(lambda n: (abspath(n),))),
        And(abc.Sequence, lambda n: all(isinstance(i, (os.PathLike, bytes, str)) for i in n),  # type: ignore[misc] # noqa: E501
            Use(lambda n: tuple(abspath(i) for i in n))),
        error=Formatter(f"'psf.psf_file' expected None or one or more path-like objects{EPILOG}")
    ),

    Optional_('ligand_atoms', default=None): Or(
        None,
        And(str, Use(lambda n: frozenset({n}))),
        And(abc.Collection, lambda n: all(isinstance(i, str) for i in n), Use(frozenset)),
        error=Formatter(f"'psf.ligand_atoms' expected None or a Collection of atoms{EPILOG}")
    ),
}, name='psf_schema', description='Schema for validating the "psf" block.')


class PSFMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`psf_schema`."""

    str_file: Union[None, bytes, str, os.PathLike, Sequence[Union[str, bytes, os.PathLike]]]
    rtf_file: Union[None, bytes, str, os.PathLike, Sequence[Union[str, bytes, os.PathLike]]]
    psf_file: Union[None, bytes, str, os.PathLike, Sequence[Union[str, bytes, os.PathLike]]]
    ligand_atoms: Optional[Collection[str]]


class PSFDict(TypedDict):
    """A typed dict represneting the output of :data:`psf_schema`."""

    str_file: Optional[Tuple[str, ...]]
    rtf_file: Optional[Tuple[str, ...]]
    psf_file: Optional[Tuple[str, ...]]
    ligand_atoms: Optional[FrozenSet[str]]


#: Schema for validating the ``"pes"`` block.
pes_schema = Schema({
    'func': Or(
        And(str, Use(import_func)),
        abc.Callable,
        error=Formatter(f"'pes.*.func' expected a callable object{EPILOG}")
    ),

    Optional_('ref', default=None): Or(
        None,
        And(abc.Sequence, Use(list)),
        error=Formatter(f"'pes.*.ref' expected a Sequence of array-likes{EPILOG}"),
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
        error=Formatter(f"'pes.*.kwargs' expected a Mapping or Sequence of Mappings{EPILOG}")
    ),

    Optional_('err_func', default=lambda: default_error_func): Or(
        And(None, Default(default_error_func, call=False)),
        And(str, Use(import_func)),
        abc.Callable,
        error=Formatter(f"'pes.*.err_func' expected a callable object{EPILOG}")
    ),
}, name='pes_schema', description='Schema for validating the "pes" block.')


class _PESMapping(TypedDict):
    func: Union[str, Callable]


class PESMapping(_PESMapping, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`pes_schema`."""

    ref: Optional[Sequence[ArrayLike]]
    kwargs: Union[None, Mapping[str, Any], Sequence[Mapping[str, Any]]]
    err_func: None | str | Callable[[ArrayLike, ArrayLike], np.number | float]


class PESDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`pes_schema`."""

    func: Callable[..., ArrayLike]
    ref: Optional[List[ArrayLike]]
    kwargs: Union[Mapping[str, Any], Tuple[Mapping[str, Any], ...]]
    err_func: Callable[[ArrayLike, ArrayLike], np.number | float]


#: Schema for validating the ``"job"`` block.
job_schema = Schema({
    Optional_('type', default=lambda: PackageManager): Or(
        And(None, Default(PackageManager, call=False)),
        And(str, Use(import_pkg)),
        And(type, pkg_subclass),
        error=Formatter(f"'job.func' expected a PackageManager class{EPILOG}")
    ),

    'molecule': Or(
        And(MultiMolecule, Use(lambda n: (n,))),
        And(str, Use(lambda n: (MultiMolecule.from_xyz(n),))),
        And(os.PathLike, Use(lambda n: (MultiMolecule.from_xyz(n),))),
        And(
            abc.Sequence,
            lambda n: all(isinstance(i, (str, os.PathLike, MultiMolecule)) for i in n),  # type: ignore[misc] # noqa: E501
            Use(lambda n: tuple(
                (i if isinstance(i, MultiMolecule) else MultiMolecule.from_xyz(i)) for i in n
            ))
        ),
        error=Formatter(f"'job.molecule' expected one or more .xyz files{EPILOG}")
    ),

    Optional_('lattice', default=None): Or(
        None,
        And(Or(str, bytes, os.PathLike), Use(lambda n: (parse_lattice(n),))),
        And(
            abc.Sequence,
            lambda n: all(isinstance(i, (str, bytes, os.PathLike)) for i in n),
            Use(lambda n: tuple(parse_lattice(i) for i in n)),
        ),
        error=Formatter(f"'job.lattice' expected one or more .cell files{EPILOG}")
    ),
}, name='job_schema', description='Schema for validating the "job" block.')


class JobMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`job_schema`."""

    type: Union[None, str, Type[PackageManagerABC]]
    molecule: Union[MultiMolecule, str, os.PathLike, Sequence[Union[MultiMolecule, str, os.PathLike]]]  # noqa: E501
    lattice: Union[
        None,
        str,
        bytes,
        os.PathLike[Any],
        Sequence[str | bytes | os.PathLike[Any]],
    ]


class JobDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`job_schema`."""

    type: Type[PackageManagerABC]
    molecule: Tuple[MultiMolecule, ...]
    lattice: None | Tuple[NDArray[np.float64], ...]


def _parse_settings_sequence(seq: Sequence[Mapping]) -> Generator[QmSettings, None, None]:
    for n in seq:
        if isinstance(n, abc.Mapping):
            yield QmSettings(n)
        else:
            raise TypeError(f"Expected a Mapping; observed type: {n.__class__.__name__}")


#: Schema for validating sub blocks within the ``"job"`` block.
sub_job_schema = Schema({
    Optional_('type', default=lambda: cp2k_mm): Or(
        And(None, Default(cp2k_mm, call=False)),
        And(str, Use(import_qm_pkg)),
        Package,
        error=Formatter(f"'job.*.type' expected a Package instance{EPILOG}")
    ),

    Optional_('settings', default=[QmSettings()]): Or(
        And(None, Default([QmSettings()])),
        And(abc.Mapping, Use(lambda n: list(_parse_settings_sequence([n])))),
        And(abc.Sequence, Use(lambda seq: list(_parse_settings_sequence(seq)))),
        error=Formatter(f"'job.*.settings' expected a Mapping{EPILOG}")
    ),

    Optional_('template', default=QmSettings): Or(
        And(None, Default(QmSettings)),
        And(str, Use(lambda n: QmSettings(import_mapping(n)))),
        And(abc.Mapping, Use(QmSettings)),
        error=Formatter(f"'job.*.template' expected a string or a Mapping{EPILOG}")
    )
}, name='sub_job_schema', description='Schema for validating sub blocks within the "job" block.')


class SubJobMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`sub_job_schema`."""

    type: Union[None, str, Type[Package]]
    settings: Union[None, Mapping, Sequence[Mapping]]
    template: Union[None, str, Mapping]


class SubJobDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`sub_job_schema`."""

    type: Type[Package]
    settings: List[QmSettings]
    template: QmSettings


#: A schema for validating the "param.validation" block.
validation_schema = Schema({
    Optional_('allow_non_existent', default=False): Or(
        And(None, Default(False)),
        bool,
        error=Formatter(f"'param.validation.allow_non_existent' expected a boolean{EPILOG}")
    ),

    Optional_('charge_tolerance', default=0.01): Or(
        And(None, Default(0.01)),
        And(supports_float, lambda n: float(n) > 0, Use(float)),
        error=Formatter(f"'param.validation.charge_tolerance' expected a positive float{EPILOG}")
    ),

    Optional_('enforce_constraints', default=False): Or(
        And(None, Default(False)),
        bool,
        error=Formatter(f"'param.validation.enforce_constraints' expected a boolean{EPILOG}")
    ),
}, name='validation_schema', description='Schema for validating the "param.validation" block.')


class ValidationMapping(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`validation_schema`."""

    allow_non_existent: Optional[bool]
    charge_tolerance: Union[None, str, bytes, SupportsFloat]
    enforce_constraints: Optional[bool]


class ValidationDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`validation_schema`."""

    allow_non_existent: bool
    charge_tolerance: float
    enforce_constraints: bool


MOVE_DEFAULT = np.array([
    0.900, 0.905, 0.910, 0.915, 0.920, 0.925, 0.930, 0.935, 0.940,
    0.945, 0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985,
    0.990, 0.995, 1.005, 1.010, 1.015, 1.020, 1.025, 1.030, 1.035,
    1.040, 1.045, 1.050, 1.055, 1.060, 1.065, 1.070, 1.075, 1.080,
    1.085, 1.090, 1.095, 1.100
], dtype=float)
MOVE_DEFAULT.setflags(write=False)


#: Schema for validating the ``"param"`` block.
param_schema = Schema({
    Optional_('type', default=lambda: ParamMapping): Or(
        And(None, Default(ParamMapping, call=False)),
        And(str, Use(import_prm)),
        And(type, prm_subclass),
        error=Formatter(f"'param.type' expected a ParamMapping class{EPILOG}")
    ),

    Optional_('func', default=lambda: np.multiply): Or(
        And(None, Default(np.multiply, call=False)),
        And(str, Use(import_func)),
        abc.Callable,
        error=Formatter(f"'param.func' expected a Callable object{EPILOG}")
    ),

    Optional_('kwargs', default=dict): Or(
        And(None, Default(dict)),
        And(abc.Mapping, lambda n: all(isinstance(k, str) for k in n.keys())),
        error=Formatter(f"'param.kwargs' expected a Mapping{EPILOG}")
    ),

    Optional_('move_range', default=MOVE_DEFAULT.copy): Or(
        And(None, Default(MOVE_DEFAULT.copy)),
        And(Or(abc.Iterable, ArrayLike), Use(array)),
        And(abc.Mapping, lambda n: {'start', 'stop', 'step', 'ratio'}.issuperset(n.keys()),
            Use(lambda n: get_move_range(**n))),
        error=Formatter(f"'param.move_range' expected a Mapping or array-like object{EPILOG}")
    ),

    Optional_('validation', default=dict): Or(
        And(None, Default(dict)),
        And(abc.Mapping, Use(dict)),
        error=Formatter(f"'param.validation' expected a dictionary{EPILOG}")
    ),
}, name='param_schema', description='Schema for validating the "param" block.')


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
    move_range: Union[None, ArrayLike, MoveRange]
    validation: Optional[ValidationMapping]


class ParamDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`param_schema`."""

    type: Type[ParamMappingABC]
    func: Callable[[float, float], float]
    kwargs: Mapping[str, Any]
    move_range: np.ndarray
    allow_non_existent: bool
    validation: ValidationMapping


#: Schema for validating the main input blocks.
main_schema = Schema({
    'param': Or(
        abc.MutableMapping,
        And(abc.Mapping, Use(dict)),
        error=Formatter(f"'param' expected a Mapping{EPILOG}")
    ),

    'pes': Or(
        abc.MutableMapping,
        And(abc.Mapping, Use(dict)),
        error=Formatter(f"'pes' expected a Mapping{EPILOG}")
    ),

    'job': Or(
        abc.MutableMapping,
        And(abc.Mapping, Use(dict)),
        error=Formatter(f"'job' expected a Mapping{EPILOG}")
    ),

    Optional_('pes_validation', default=dict): Or(
        abc.MutableMapping,
        And(abc.Mapping, Use(dict)),
        error=Formatter(f"'pes_validation' expected a Mapping{EPILOG}")
    ),

    Optional_('monte_carlo', default=dict): Or(
        And(None, Default(dict)),
        abc.MutableMapping,
        And(abc.Mapping, Use(dict)),
        error=Formatter(f"'monte_carlo' expected a Mapping{EPILOG}")
    ),

    Optional_('phi', default=dict): Or(
        And(None, Default(dict)),
        abc.MutableMapping,
        And(abc.Mapping, Use(dict)),
        error=Formatter(f"'phi' expected a Mapping{EPILOG}")
    ),

    Optional_('psf', default=dict): Or(
        And(None, Default(dict)),
        abc.MutableMapping,
        And(abc.Mapping, Use(dict)),
        error=Formatter(f"'psf' expected a Mapping{EPILOG}")
    )
}, name='main_schema', description='Schema for validating the main input blocks.')


class _MainMapping(TypedDict):
    param: ParamMapping_
    pes: Mapping[str, PESMapping]
    job: JobMapping


class MainMapping(_MainMapping, total=False):
    """A :class:`~typing.TypedDict` representing the input of :data:`main_schema`."""

    pes_validation: Mapping[str, PESMapping]
    monte_carlo: Optional[MCMapping]
    phi: Optional[PhiMapping]
    psf: Optional[PSFMapping]


class MainDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`main_schema`."""

    param: ParamMapping_
    pes: Dict[str, PESMapping]
    pes_validation: Dict[str, PESMapping]
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
