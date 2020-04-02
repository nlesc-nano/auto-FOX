"""
FOX.armc.sanitization
=====================

A module for parsing and sanitizing ARMC settings.

"""

import os
import copy
from pathlib import Path
from os.path import join, isfile, abspath
from collections import abc
from itertools import chain
from typing import (
    Union, Iterable, Tuple, Optional, Mapping, Any, MutableMapping, Hashable,
    Dict, TYPE_CHECKING, Generator, Callable, List, Collection, TypeVar, overload
)

import numpy as np
import pandas as pd

from scm.plams import Settings, Molecule

from .mc_post_process import AtomsFromPSF
from .schemas import (
    validate_phi, validate_pes, validate_monte_carlo, validate_psf,
    validate_job, validate_sub_job, validate_param, validate_main,
    PESDict, PESMapping, PhiMapping, MainMapping, ParamMapping_, MCMapping,
    PSFMapping, JobMapping
)

from ..type_hints import Literal, TypedDict
from ..io.read_psf import PSFContainer, overlay_str_file, overlay_rtf_file
from ..classes import MultiMolecule
from ..functions.cp2k_utils import set_keys
from ..functions.molecule_utils import fix_bond_orders, residue_argsort
from ..functions.utils import dict_to_pandas, get_atom_count, split_dict

if TYPE_CHECKING:
    from .package_manager import PackageManager
    from .param_mapping import ParamMapping
    from .phi_updater import PhiUpdater
    from .monte_carlo import MonteCarloABC
else:
    from ..type_alias import PackageManager, ParamMapping, PhiUpdater, MonteCarloABC

__all__ = ['init_armc_sanitization']

KT = TypeVar('KT', bound=Hashable)
MT = TypeVar('MT', bound=Mapping[Any, Any])


class RunDict(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :func:`run_armc`."""

    path: Union[str, os.PathLike]
    folder: Union[str, os.PathLike]
    logfile: Union[str, os.PathLike]
    restart: bool
    psf: Union[None, List[PSFContainer], Tuple[Union[str, os.PathLike], ...]]
    guess: Optional[Mapping[str, Mapping]]


def init_armc_sanitization(input_dict: MainMapping) -> Tuple[MonteCarloABC, RunDict]:
    """Initialize the armc input settings sanitization.

    Parameters
    ----------
    input_dict : :class:`dict`
        A dictionary containing all ARMC settings.

    Returns
    -------
    :class:`ARMC` and :class:`dict`
        A Settings instance suitable for ARMC initialization.

    """
    dct = validate_main(copy.deepcopy(input_dict))

    # Construct an ARMC instance
    phi = get_phi(dct['phi'])
    package, mol_list = get_package(dct['job'])
    param, _param, _param_frozen = get_param(dct['param'])
    mc, run_kwargs = get_armc(dct['monte_carlo'], package, param, phi, mol_list)

    # Update the job Settings
    if _param_frozen is not None:
        package.update_settings(list(prm_iter(_param_frozen)), new_keys=True)
    package.update_settings(list(prm_iter(_param)), new_keys=True)

    # Handle psf stuff
    run_kwargs['psf'] = psf_list = get_psf(dct['psf'], mol_list)
    update_count(param, psf=psf_list, mol=mol_list)
    if psf_list is not None:
        mc.pes_post_process = [AtomsFromPSF.from_psf(*psf_list)]
        workdir = Path(run_kwargs['path']) / run_kwargs['folder']
        _update_psf_settings(package.values(), workdir)

    # Add PES evaluators
    pes = get_pes(dct['pes'])
    for name, kwargs in pes.items():
        mc.add_pes_evaluator(name, **kwargs)

    return mc, run_kwargs


def get_phi(dct: PhiMapping) -> PhiUpdater:
    """Construct a :class:`PhiUpdater` instance from **dct**."""
    phi_dict = validate_phi(dct)
    phi_type = phi_dict.pop('type')  # type: ignore
    kwargs = phi_dict.pop('kwargs')  # type: ignore
    return phi_type(**phi_dict, **kwargs)


def get_package(dct: JobMapping) -> Tuple[PackageManager, Tuple[MultiMolecule, ...]]:
    """Construct a :class:`PackageManager` instance from **dct**."""
    _sub_pkg_dict: Dict[str, Any] = split_dict(dct, keep_keys={'type', 'molecule'})

    job_dict = validate_job(dct)
    mol_list = [mol.as_Molecule(mol_subset=0)[0] for mol in job_dict['molecule']]

    data: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in _sub_pkg_dict.items():
        data[k] = []
        for mol in mol_list:
            kwargs = validate_sub_job(v)
            kwargs['molecule'] = mol.copy()

            pkg_name = kwargs['type'].pkg_name
            kwargs['settings'].specific[pkg_name].soft_update(kwargs.pop('template'))
            data[k].append(kwargs)

    pkg_type = job_dict['type']
    return pkg_type(data), job_dict['molecule']


def get_param(dct: ParamMapping_) -> Tuple[ParamMapping, dict, dict]:
    """Construct a :class:`ParamMapping` instance from **dct**."""
    _prm_dict = dct
    _sub_prm_dict = split_dict(_prm_dict, keep_keys={'type', 'move_range', 'func', 'kwargs'})

    prm_dict = validate_param(_prm_dict)
    kwargs = prm_dict.pop('kwargs')
    data = _get_param_df(_sub_prm_dict)
    _sub_prm_dict_frozen = _get_prm_frozen(_sub_prm_dict)

    param_type = prm_dict.pop('type')  # type: ignore
    return param_type(data, **prm_dict, **kwargs), _sub_prm_dict, _sub_prm_dict_frozen


def get_pes(dct: Mapping[str, PESMapping]) -> Dict[str, PESDict]:
    """Construct a :class:`dict` with PES-descriptor workflows."""
    return {k: validate_pes(v) for k, v in dct.items()}


def get_armc(dct: MCMapping,
             package_manager: PackageManager,
             param: ParamMapping,
             phi: PhiUpdater,
             mol: Iterable[MultiMolecule]) -> Tuple[MonteCarloABC, RunDict]:
    """Construct an :class:`ARMC` instance from **dct**."""
    mc_dict = validate_monte_carlo(dct)

    pop_keys = ('path', 'folder', 'logfile')
    kwargs = {k: mc_dict.pop(k) for k in pop_keys}  # type: ignore

    workdir = os.path.join(kwargs['path'], kwargs['folder'])
    logfile = kwargs['logfile']
    hdf5 = mc_dict['hdf5_file']
    if not os.path.isdir(os.path.dirname(logfile)):
        kwargs['logfile'] = os.path.join(workdir, logfile)
    if not os.path.isdir(os.path.dirname(hdf5)):
        mc_dict['hdf5_file'] = os.path.join(workdir, hdf5)

    mc_type = mc_dict.pop('type')  # type: ignore
    return mc_type(phi=phi, param=param, package_manager=package_manager,
                   molecule=mol, **mc_dict), kwargs


def get_psf(dct: PSFMapping, mol_list: Iterable[MultiMolecule]
            ) -> Optional[List[PSFContainer]]:
    """Construct a list of :class:`PSFContainer` instances from **dct**."""
    psf_dict = validate_psf(dct)

    atoms = psf_dict.get('ligand_atoms')
    mol_list_ = [mol.as_Molecule(mol_subset=0)[0] for mol in mol_list]

    if psf_dict['psf_file'] is not None:
        return [PSFContainer.read(file) for file in psf_dict['psf_file']]

    elif psf_dict['rtf_file'] is not None:
        return _generate_psf(psf_dict['rtf_file'], mol_list_, ligand_atoms=atoms, mode='rtf')

    elif psf_dict['str_file'] is not None:
        return _generate_psf(psf_dict['str_file'], mol_list_, ligand_atoms=atoms, mode='str')

    else:
        return None


Mode = Literal['str', 'rtf']


def _generate_psf(file_list: Iterable[Union[str, os.PathLike]],
                  mol_list: Iterable[Molecule],
                  ligand_atoms: Optional[Collection[str]] = None,
                  mode: Mode = 'rtf') -> List[PSFContainer]:
    ret = []
    for file, mol in zip(file_list, mol_list):
        if ligand_atoms is not None:
            atom_subset = [at for at in mol if at.symbol in ligand_atoms]
            mol.guess_bonds(atom_subset=atom_subset)

        # Create a and sanitize a plams molecule
        res_list = residue_argsort(mol, concatenate=False)
        _assign_residues(mol, res_list)

        # Initialize and populate the psf instance
        psf = PSFContainer()
        psf.generate_bonds(mol)
        psf.generate_angles(mol)
        psf.generate_dihedrals(mol)
        psf.generate_impropers(mol)
        psf.generate_atoms(mol)
        psf.charge = 0.0

        # Overlay the PSFContainer instance with either the .rtf or .str file
        if mode == 'str':
            overlay_str_file(psf, file)
        elif mode == 'rtf':
            overlay_rtf_file(psf, file)
        else:
            raise ValueError(repr(mode))
        ret.append(psf)
    return ret


def _update_psf_settings(job_lists: Iterable[Iterable[dict]],
                         workdir: Union[str, os.PathLike]) -> None:
    """Set the .psf path in all job settings."""
    for job_list in job_lists:
        for i, job in enumerate(job_list):
            job['settings'].psf = os.path.join(workdir, f'mol.{i}.psf')


NestedDict = Mapping[KT, Union[MT, Iterable[MT]]]
PrmTuple = Tuple[str, str, str, float]


def prm_iter(dct: NestedDict) -> Generator[Tuple[KT, MT], None, None]:
    """Create a an iterator yielding individual parameter dictionaries."""
    for key_alias, _dct_list in dct.items():

        # Ensure that we're dealing with a list of dicts
        if isinstance(_dct_list, abc.Mapping):
            dct_list: Iterable[MT] = [_dct_list]
        else:
            dct_list = _dct_list

        for sub_dict in dct_list:
            yield (key_alias, sub_dict)


def _get_param_df(dct: Mapping[str, Any]) -> pd.DataFrame:
    """Construct a DataFrame for :class:`ParamMapping`."""
    columns = ['key', 'param_type', 'atoms', 'param']
    data = _get_param(dct)

    df = pd.DataFrame(data, columns=columns)
    df.set_index(['key', 'param_type', 'atoms'], inplace=True)
    return df


def _get_param(dct: Mapping[str, Union[Mapping, Iterable[Mapping]]]
               ) -> Generator[PrmTuple, None, None]:
    """Create a generator yielding DataFrame rows for :class:`ParamMapping`."""
    ignore_keys = {'frozen', 'constraints', 'param', 'unit', 'guess'}

    dct_iterator = prm_iter(dct)
    for key, sub_dct in dct_iterator:
        param = sub_dct['param']
        for atoms, value in sub_dct.items():
            if atoms in ignore_keys:
                continue
            yield key, param, atoms, value


def _get_prm_frozen(dct: Mapping[str, Union[MutableMapping, Iterable[MutableMapping]]]
                    ) -> Optional[Dict[str, List[dict]]]:
    """Extract  a generator yielding DataFrame rows for :class:`ParamMapping`."""
    ret: Dict[str, List[dict]] = {}

    dct_iterator = prm_iter(dct)
    for key_alias, sub_dict in dct_iterator:
        try:
            frozen = sub_dict.pop('frozen')
        except KeyError:
            continue

        frozen['param'] = sub_dict['param']
        frozen['unit'] = sub_dict.get('unit')
        try:
            ret[key_alias].append(frozen)
        except KeyError:
            ret[key_alias] = [frozen]
    return ret if ret else None


@overload
def update_count(param: ParamMapping, psf: Iterable[PSFContainer], mol: None) -> None: ...
@overload
def update_count(param: ParamMapping, psf: None, mol: Iterable[MultiMolecule]) -> None: ...
def update_count(param, psf=None, mol=None):  # noqa: E302
    """Assign atomc-ounts to the passed :class:`ParamMapping`."""
    # Construct a gener
    if psf is not None:
        count_iter = (pd.value_counts(p.atom_type) for p in psf)
    elif mol is not None:
        count_iter = (m.atoms for m in mol)
    else:
        raise TypeError("'psf' and 'mol' cannot be both 'None'")

    prm_count = param['count']
    at_sequence = [atoms.split() for *_, atoms in prm_count.index]
    for count in count_iter:
        data = get_atom_count(at_sequence, count)
        series = pd.Series({k: v for k, v in zip(prm_count.index, data) if v is not None},
                           name='unit')
        prm_count.update(series)


def _assign_residues(plams_mol: Molecule, res_list: Iterable[Iterable[int]]) -> None:
    fix_bond_orders(plams_mol)
    res_name = 'COR'
    for i, j_list in enumerate(res_list, 1):
        for j in j_list:
            j += 1
            plams_mol[j].properties.pdb_info.ResidueNumber = i
            plams_mol[j].properties.pdb_info.ResidueName = res_name
        res_name = 'LIG'
