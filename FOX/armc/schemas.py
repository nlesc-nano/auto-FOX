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
from typing import (overload, Any, SupportsInt, SupportsFloat, Type, Mapping,
                    Callable, Union, Optional, Tuple)

import numpy as np
from schema import And, Or, Schema, Use, Optional as Optional_

from scm.plams import Cp2kJob, SingleJob, Settings

from .armc import ARMC
from .monte_carlo import MonteCarloABC
from .workflow_manager import WorkflowManager, WorkflowManagerABC
from .phi_updater import PhiUpdater, PhiUpdaterABC
from .param_mapping import ParamMapping, ParamMappingABC
from ..type_hints import Literal, SupportsArray, TypedDict, NDArray
from ..classes import MultiMolecule
from ..functions.utils import str_to_callable, _get_move_range

__all__ = [
    'validate_phi', 'validate_monte_carlo', 'validate_psf', 'validate_pes',
    'validate_job', 'validate_sub_job', 'validate_param'
]


@overload
def supports_float(value: SupportsFloat) -> Literal[True]: ...
@overload
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
@overload
def supports_int(value: Any) -> bool: ...
def supports_int(value):  # noqa: E302
    """Check if a int-like object has been passed (:data:`~typing.SupportsInt`)."""
    # floats that can be exactly represented by an integer are also fine
    try:
        value.__int__()
        return float(value).is_integer()
    except Exception:
        return False


#: Schema for validating the ``"phi"`` block.
phi_schema = Schema({
    Optional_('type', default=lambda: PhiUpdater): Or(
        And(str, Use(str_to_callable)),
        And(type, lambda n: issubclass(n, PhiUpdaterABC))
    ),

    Optional_('phi', default=1.0):
        And(supports_float, Use(float)),

    Optional_('gamma', default=2.0):
        And(supports_float, Use(float)),

    Optional_('a_target', default=0.25):
        And(supports_float, lambda n: 0 < float(n) <= 1, Use(float)),

    Optional_('func', default=lambda: np.add): Or(
        And(str, Use(str_to_callable)),
        And(abc.Callable)
    ),

    Optional_('kwargs', default=dict):
        And(abc.Mapping, lambda dct: all(isinstance(k, str) for k, _ in dct.items()))
})


class PhiDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`phi_schema`."""

    type: Type[PhiUpdaterABC]
    phi: float
    gamma: float
    a_target: float
    func: Callable[[float, float], float]
    kwargs: Mapping[str, Any]


#: Schema for validating the ``"monte_carlo"`` block.
mc_schema = Schema({
    Optional_('type', default=lambda: ARMC): Or(
        And(str, Use(str_to_callable)),
        And(type, lambda n: issubclass(n, MonteCarloABC))
    ),

    Optional_('iter_len', default=50000): Or(
        And(supports_int, lambda n: int(n) > 0, Use(int))
    ),

    Optional_('sub_iter_len', default=100): Or(
        And(supports_int, lambda n: int(n) > 0, Use(int))
    ),

    Optional_('hdf5_file', default='armc.hdf5'): Or(
        Or(str, os.PathLike)
    ),

    Optional_('path', default=os.getcwd): Or(
        Or(str, os.PathLike)
    ),

    Optional_('folder', default='MM_MD_workdir'): Or(
        Or(str, os.PathLike)
    ),

    Optional_('keep_files', default=True):
        bool
})


class MCDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`mc_schema`."""

    type: Type[MonteCarloABC]
    iter_len: int
    sub_iter_len: int
    hdf5_file: Union[str, os.PathLike]
    path: Union[str, os.PathLike]
    folder: Union[str, os.PathLike]
    keep_files: bool


#: Schema for validating the ``"psf"`` block.
psf_schema = Schema({
    Optional_('str_file', default=None): Or(
        None,
        And(str, Use(lambda n: (n,))),
        And(os.PathLike, Use(lambda n: (n,))),
        And(abc.Sequence, lambda x: all(isinstance(i, (os.PathLike, str)) for i in x), Use(tuple))
    ),

    Optional_('rtf_file', default=None): Or(
        None,
        And(str, Use(lambda n: (n,))),
        And(os.PathLike, Use(lambda n: (n,))),
        And(abc.Sequence, lambda x: all(isinstance(i, (os.PathLike, str)) for i in x), Use(tuple))
    ),

    Optional_('psf_file', default=None): Or(
        None,
        And(str, Use(lambda n: (n,))),
        And(os.PathLike, Use(lambda n: (n,))),
        And(abc.Sequence, lambda x: all(isinstance(i, (os.PathLike, str)) for i in x), Use(tuple))
    ),

    Optional_('ligand_atoms', default=None): Or(
        None,
        And(str, Use(lambda n: (n,))),
        And(abc.Sequence, lambda x: all(isinstance(i, str) for i in x), Use(tuple))
    ),
})


class PSFDict(TypedDict):
    """A typed dict represneting the output of :data:`psf_schema`."""

    str_file: Optional[Tuple[Union[str, os.PathLike], ...]]
    rtf_file: Optional[Tuple[Union[str, os.PathLike], ...]]
    psf_file: Optional[Tuple[Union[str, os.PathLike], ...]]
    ligand_atoms: Optional[Tuple[str, ...]]


#: Schema for validating the ``"pes"`` block.
pes_schema = Schema({
    'func': Or(
        And(str, Use(str_to_callable)),
        abc.Callable
    ),

    Optional_('kwargs', default=dict): Or(
        And(abc.Mapping, lambda dct: all(isinstance(k, str) for k, _ in dct.items())),
        And(
            abc.Sequence,
            lambda n: all(isinstance(i, abc.Mapping) for i in n),
            lambda n: all(isinstance(k, str) for dct in n for k, _ in dct.items()),
            Use(tuple)
        )
    )
})


class PESDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`pes_schema`."""

    func: Callable
    kwargs: Union[Mapping[str, Any], Tuple[Mapping[str, Any], ...]]


#: Schema for validating the ``"job"`` block.
job_schema = Schema({
    Optional_('type', default=lambda: WorkflowManager): Or(
        And(str, Use(str_to_callable)),
        And(type, lambda n: issubclass(n, WorkflowManagerABC))
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
        )
    )
})


class JobDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`job_schema`."""

    type: Type[WorkflowManagerABC]
    molecule: Tuple[MultiMolecule, ...]


#: Schema for validating sub blocks within the ``"pes"`` block.
sub_job_schema = Schema({
    Optional_('job_type', default=lambda: Cp2kJob): Or(
        And(str, Use(str_to_callable)),
        And(type, lambda n: issubclass(n, SingleJob))
    ),

    Optional_('name', default='plamsjob'):
        str,

    'settings':
        And(abc.Mapping, Use(Settings))
})


class SubJobDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`sub_job_schema`."""

    job_type: Type[SingleJob]
    name: str
    settings: Settings


#: Schema for validating the ``"param"`` block.
param_schema = Schema({
    Optional_('type', default=lambda: ParamMapping): Or(
        And(str, Use(str_to_callable)),
        And(type, lambda n: issubclass(n, ParamMappingABC))
    ),

    Optional_('func', default=lambda: np.multiply): Or(
        And(str, Use(str_to_callable)),
        And(abc.Callable)
    ),

    Optional_('kwargs', default=dict):
        And(abc.Mapping, lambda dct: all(isinstance(k, str) for k, _ in dct.items())),

    Optional_('move_range', default=None): Or(
        And(abc.Sequence, Use(lambda n: np.asarray(n, dtype=float))),
        And(SupportsArray, Use(lambda n: np.asarray(n, dtype=float))),
        And(abc.Mapping, lambda n: {'start', 'stop', 'step'} == n.keys(), Use(_get_move_range))
    )

})


class ParamDict(TypedDict):
    """A :class:`~typing.TypedDict` representing the output of :data:`param_schema`."""

    type: Type[ParamMappingABC]
    func: Callable[[float, float], float]
    kwargs: Mapping[str, Any]
    move_range: NDArray[float]


def validate_phi(mapping: Mapping[str, Any]) -> PhiDict:
    """Validate the ``"phi"`` block."""
    return phi_schema.validate(mapping)


def validate_monte_carlo(mapping: Mapping[str, Any]) -> MCDict:
    """Validate the ``"monte_carlo"`` block."""
    return mc_schema.validate(mapping)


def validate_psf(mapping: Mapping[str, Any]) -> PSFDict:
    """Validate the ``"psf"`` block."""
    return psf_schema.validate(mapping)


def validate_pes(mapping: Mapping[str, Any]) -> PESDict:
    """Validate the ``"pes"`` block."""
    return pes_schema.validate(mapping)


def validate_job(mapping: Mapping[str, Any]) -> JobDict:
    """Validate the ``"job"`` block."""
    return job_schema.validate(mapping)


def validate_sub_job(mapping: Mapping[str, Any]) -> SubJobDict:
    """Validate sub-blocks within the ``"job"`` block."""
    return sub_job_schema.validate(mapping)


def validate_param(mapping: Mapping[str, Any]) -> ParamDict:
    """Validate the ``"param"`` block."""
    return param_schema.validate(mapping)


__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autodata='\n'.join(f'.. autofunction:: {i}' for i in __all__)
)
