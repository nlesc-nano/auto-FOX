"""
FOX.armc.sanitization
=====================

A module for parsing and sanitizing ARMC settings.

"""

import os
from pathlib import Path
from os.path import join, isfile, abspath
from collections import abc
from typing import (Union, Iterable, Tuple, Optional, Mapping, Any, MutableMapping,
                    Dict, TYPE_CHECKING, Generator, Callable, List, Collection)

import numpy as np
import pandas as pd

from scm.plams import Settings, Molecule

from .mc_post_process import AtomsFromPSF
from .schemas import (validate_phi, validate_pes, validate_monte_carlo, validate_psf,
                      validate_job, validate_sub_job, validate_param, PESDict, validate_main)

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
    from .armc import ARMC
else:
    from ..type_alias import PackageManager, ParamMapping, PhiUpdater, ARMC

__all__ = ['init_armc_sanitization']

ValidKeys = Literal['param', 'psf', 'pes', 'job', 'monte_carlo', 'phi']
InputMapping = Mapping[ValidKeys, Mapping[str, Any]]


class PesMapping(TypedDict):
    """A :class:`~typing.TypedDict` representing the input of :func:`get_pes`."""

    func: Union[str, Callable]
    kwargs: Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]


class RunDict(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :func:`run_armc`."""

    path: Union[str, os.PathLike]
    folder: Union[str, os.PathLike]
    logfile: Union[str, os.PathLike]
    restart: bool
    psf: Union[None, List[PSFContainer], Tuple[Union[str, os.PathLike], ...]]
    guess: Optional[Mapping[str, Mapping]]


def init_armc_sanitization(input_dict: InputMapping):
    """Initialize the armc input settings sanitization.

    Parameters
    ----------
    dct : dict
        A dictionary containing all ARMC settings.

    Returns
    -------
    |plams.Settings|_:
        A Settings instance suitable for ARMC initialization.

    """
    dct = validate_main(input_dict)

    # Construct an ARMC instance
    phi = get_phi(dct['phi'])
    package, mol_list = get_package(dct['job'])
    param = get_param(dct['param'])
    mc, run_kwargs = get_armc(dct['monte_carlo'], package, param, phi, mol_list)

    # Handle psf stuff
    run_kwargs['psf'] = psf_list = get_psf(dct['psf'], mol_list)
    if psf_list is not None:
        mc.pes_post_process = [AtomsFromPSF.from_psf(*psf_list)]
        workdir = Path(run_kwargs['path']) / run_kwargs['folder']
        for job_list in mc.package_manager.values():
            for i, job in enumerate(job_list):
                job['settings'].psf = workdir / f'mol.{i}.psf'

    # Add PES evaluators
    pes = get_pes(dct['pes'])
    for name, kwargs in pes.items():
        mc.add_pes_evaluator(name, **kwargs)

    # import pdb; pdb.set_trace()
    return mc, run_kwargs


def get_phi(dct: Mapping[str, Any]) -> PhiUpdater:
    """Construct a :class:`PhiUpdater` instance from **dct**."""
    phi_dict = validate_phi(dct)
    phi_type = phi_dict.pop('type')  # type: ignore
    return phi_type(**phi_dict)


def get_package(dct: MutableMapping[str, Any]
                ) -> Tuple[PackageManager, Tuple[MultiMolecule, ...]]:
    """Construct a :class:`PackageManager` instance from **dct**."""
    _job_dict = dct
    _sub_job_dict = split_dict(_job_dict, keep_keys={'type', 'molecule'})

    job_dict = validate_job(_job_dict)
    mol_list = [mol.as_Molecule(mol_subset=0)[0] for mol in job_dict['molecule']]

    data: Dict[str, List[Dict[str, Any]]] = {}
    for k, v in _sub_job_dict.items():
        data[k] = []
        for mol in mol_list:
            kwargs = validate_sub_job(v)
            kwargs['molecule'] = mol.copy()
            kwargs['settings'].soft_update(kwargs.pop('template'))
            data[k].append(kwargs)

    job_type = job_dict['type']
    return job_type(data), job_dict['molecule']


def get_param(dct: MutableMapping[str, Any]) -> ParamMapping:
    """Construct a :class:`ParamMapping` instance from **dct**."""
    _prm_dict = dct
    _sub_prm_dict = split_dict(_prm_dict, keep_keys={'type', 'move_range', 'func', 'kwargs'})

    prm_dict = validate_param(_prm_dict)
    data = _get_param_df(_sub_prm_dict)
    param_type = prm_dict.pop('type')  # type: ignore
    return param_type(data, **prm_dict)


def get_pes(dct: Mapping[str, PesMapping]) -> Dict[str, PESDict]:
    """Construct a :class:`dict` with PES-descriptor workflows."""
    return {k: validate_pes(v) for k, v in dct.items()}


def get_armc(dct: MutableMapping[str, Any],
             package_manager: PackageManager,
             param: ParamMapping,
             phi: PhiUpdater,
             mol: Iterable[MultiMolecule]) -> Tuple[ARMC, RunDict]:
    """Construct an :class:`ARMC` instance from **dct**."""
    mc_dict = validate_monte_carlo(dct)

    pop_keys = ('path', 'folder', 'logfile')
    kwargs = {k: mc_dict.pop(k) for k in pop_keys}  # type: ignore

    mc_type = mc_dict.pop('type')  # type: ignore
    return mc_type(phi=phi, param=param, package_manager=package_manager,
                   molecule=mol, **mc_dict), kwargs


def get_psf(dct: Mapping[str, Any], mol_list: Iterable[MultiMolecule]
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
        ret.append(psf)
    return ret


def _get_param_df(dct: Mapping[str, Any]) -> pd.DataFrame:
    """Construct a DataFrame for :class:`ParamMapping`."""
    columns = ['param_type', 'atoms', 'param']
    data = _prm_iter(dct)

    df = pd.DataFrame(data, columns=columns)
    df.set_index(['param_type', 'atoms'], inplace=True)
    return df


PrmTuple = Tuple[str, str, float]


def _prm_iter(dct: Mapping[str, Union[Mapping, Iterable[Mapping]]]
              ) -> Generator[PrmTuple, None, None]:
    """Create a generator yielding DataFrame rows for :class:`ParamMapping`."""
    ignore_keys = {'frozen', 'constraints', 'param', 'unit', 'guess'}

    for key_alias, _dct_list in dct.items():

        # Ensure that we're dealing with a list of dicts
        if isinstance(_dct_list, abc.Mapping):
            dct_list: Iterable[Mapping] = [_dct_list]
        else:
            dct_list = _dct_list

        # Traverse the list of dicts
        for sub_dict in dct_list:
            param = sub_dict['param']
            for atoms, value in sub_dict.items():
                if atoms in ignore_keys:
                    continue
                yield param, atoms, value


def _parse_param(s: Settings, job: str) -> None:
    """Reshape and post-process the ``"param"`` block in the validated ARMC settings.

    Parameters
    ----------
    s : |plams.Settings|_
        A Settings instance containing all ARMC settings.

    See Also
    --------
    :func:`.reshape_settings`:
        General function for reshaping and post-processing validated ARMC settings.

    """
    param = s.param
    md_settings = s.md_settings
    path = job['path']

    if param.prm_file is None:
        md_settings.input.force_eval.mm.forcefield.parmtype = 'OFF'
        del param.prm_file
    else:
        prm_file = param.pop('prm_file')
        prm_file_ = abspath(prm_file) if isfile(abspath(prm_file)) else join(path, prm_file)
        md_settings.input.force_eval.mm.forcefield.parm_file_name = prm_file_

    # Create a copy of s.param with just all frozen settings
    prm_frozen = Settings()
    job['guess'] = {}
    for k, v in param.items():
        if 'guess' in v:
            job['guess'][k] = {'mode': v.guess, 'frozen': False}
            del v.guess

        if 'frozen' not in v:
            continue
        if 'keys' in v:
            prm_frozen[k]['keys'] = v['keys']
        if 'unit' in v:
            prm_frozen[k].unit = v.unit
        if 'guess' in v.frozen:
            job['guess'][k] = {'mode': v.frozen.guess, 'frozen': True}
            del v.frozen.guess
        prm_frozen[k].update(v.pop('frozen'))

    if prm_frozen:
        df_frozen = dict_to_pandas(prm_frozen, 'param')
        set_keys(md_settings, df_frozen)

    param = s.param = dict_to_pandas(s.param, 'param')
    param['param_old'] = np.nan
    set_keys(md_settings, param)
    if 'constraints' not in param.columns:
        param['constraints'] = None
    param['count'] = 0


def _assign_residues(plams_mol: Molecule, res_list: Iterable[Iterable[int]]) -> None:
    fix_bond_orders(plams_mol)
    res_name = 'COR'
    for i, j_list in enumerate(res_list, 1):
        for j in j_list:
            j += 1
            plams_mol[j].properties.pdb_info.ResidueNumber = i
            plams_mol[j].properties.pdb_info.ResidueName = res_name
        res_name = 'LIG'
