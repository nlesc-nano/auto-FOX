"""
FOX.armc.sanitization
=====================

A module for parsing and sanitizing ARMC settings.

"""

import os
import functools
from os.path import join, isfile, abspath
from collections import abc
from typing import (Union, Iterable, Tuple, Optional, Mapping, Any, MutableMapping,
                    Dict, TYPE_CHECKING, Hashable, TypeVar, Generator, Callable)

import numpy as np
import pandas as pd

from scm.plams import Settings, Molecule
from qmflows.cp2k_utils import CP2K_KEYS_ALIAS

from .mc_post_process import AtomsFromPSF
from .schemas import (validate_phi, validate_pes, validate_monte_carlo, validate_psf,
                      validate_job, validate_sub_job, validate_param, PESDict)

from ..type_hints import Literal, TypedDict
from ..io.read_psf import PSFContainer, overlay_str_file, overlay_rtf_file
from ..classes import MultiMolecule
from ..functions.cp2k_utils import set_keys, set_subsys_kind
from ..functions.molecule_utils import fix_bond_orders
from ..functions.utils import (get_template, dict_to_pandas, get_atom_count,
                               _get_move_range, split_dict)

if TYPE_CHECKING:
    from .workflow_manager import WorkflowManager
    from .param_mapping import ParamMapping
    from .phi_updater import PhiUpdater
    from .armc import ARMC
else:
    from ..type_alias import WorkflowManager, ParamMapping, PhiUpdater, ARMC

__all__ = ['init_armc_sanitization']

ValidKeys = Literal['param', 'psf', 'pes', 'job', 'monte_carlo', 'phi']
InputMapping = Mapping[ValidKeys, MutableMapping[str, Any]]


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
    psf: Optional[Tuple[PSFContainer, ...]]
    guess: Optional[Mapping[str, Mapping]]


def init_armc_sanitization(dct: InputMapping):
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
    # Construct an ARMC instance
    phi = get_phi(dct['phi'])
    workflow, mol_list = get_workflow(dct['job'])
    param = get_param(dct['param'])
    mc, run_kwargs = get_armc(dct['monte_carlo'], workflow, param, phi, mol_list)

    # Handle psf stuff
    run_kwargs['psf'] = get_psf(dct['psf'])

    # Add PES evaluators
    pes = get_pes(dct['pes'])
    for name, kwargs in pes.items():
        mc.add_pes_evaluator(name, **kwargs)

    return mc, run_kwargs

    # Validate, post-process and return
    s = validate(s_inp)
    _parse_move(s)
    _parse_armc(s)
    job: Settings = _parse_job(s)
    pes: Settings = _parse_pes(s)

    _parse_param(s, job)
    job.psf = _parse_psf(s, job.path)
    _parse_preopt(s)

    if job.get('psf', None) is not None:
        s['pes_post_process'] = [AtomsFromPSF.from_psf(*job['psf'])]
    return s, pes, job


def get_phi(dct: Mapping[str, Any]) -> PhiUpdater:
    """Construct a :class:`PhiUpdater` instance from **dct**."""
    phi_dict = validate_phi(dct['phi'])
    phi_type = phi_dict.pop('type')  # type: ignore
    return phi_type(**phi_dict)


def get_workflow(dct: MutableMapping[str, Any]
                 ) -> Tuple[WorkflowManager, Tuple[MultiMolecule, ...]]:
    """Construct a :class:`WorkflowManager` instance from **dct**."""
    _job_dict = dct
    _sub_job_dict = split_dict(_job_dict, keep_keys={'type', 'molecule'})

    job_dict = validate_job(_job_dict)
    mol_list = [mol.as_Molecule(mol_subset=0)[0] for mol in job_dict['molecule']]

    data = {}
    for k, v in _sub_job_dict.items():
        sub_job_dict = validate_sub_job(v)
        singleob_type = sub_job_dict.pop('type')  # type: ignore
        data[k] = [singleob_type(molecule=mol, **sub_job_dict) for mol in mol_list]

    job_type = job_dict['type']
    return job_type(data, post_process=None), job_dict['molecule']


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


def get_armc(dct: MutableMapping[str, Any], workflow_manager: WorkflowManager,
             param: ParamMapping, phi: PhiUpdater, mol: Iterable[MultiMolecule]
             ) -> Tuple[ARMC, RunDict]:
    """Construct an :class:`ARMC` instance from **dct**."""
    mc_dict = validate_monte_carlo(dct)

    pop_keys = ('path', 'folder', 'logfile')
    kwargs = {k: mc_dict.pop(k) for k in pop_keys}  # type: ignore

    mc_type = mc_dict.pop('type')  # type: ignore
    return mc_type(phi=phi, param=param, workflow_manager=workflow_manager,
                   molecule=mol, **mc_dict), kwargs


def get_psf(dct: Mapping[str, Any]):
    psf_dict = validate_psf(dct)
    return psf_dict


def _get_param_df(dct: Mapping[str, Any]) -> pd.DataFrame:
    """Construct a DataFrame for :class:`ParamMapping`."""
    columns = ['param_type', 'atoms', 'param', 'unit', 'key_path']
    data = _prm_iter(dct)

    df = pd.DataFrame(data, columns=columns)
    df.set_index(['param_type', 'atoms'], inplace=True)
    return df


PrmTuple = Tuple[str, str, float, str, Tuple[str, ...]]


def _prm_iter(dct: Mapping[str, Union[Mapping, Iterable[Mapping]]]
              ) -> Generator[PrmTuple, None, None]:
    """Create a generator yielding DataFrame rows for :class:`ParamMapping`."""
    ignore_keys = {'frozen', 'constraints', 'param', 'unit', 'guess'}

    for key_alias, _dct_list in dct.items():
        key_path = CP2K_KEYS_ALIAS[key_alias]

        # Ensure that we're dealing with a list of dicts
        if isinstance(_dct_list, abc.Mapping):
            dct_list: Iterable[Mapping] = [_dct_list]
        else:
            dct_list = _dct_list

        # Traverse the list of dicts
        for sub_dict in dct_list:
            param = sub_dict['param']
            unit = sub_dict.get('unit', None)
            for atoms, value in sub_dict.items():
                if atoms in ignore_keys:
                    continue
                yield param, atoms, value, unit, key_path



def validate(s: Settings) -> Settings:
    """Validate all settings in **s** using schema_.

    The PLAMS Settings instance containing all input settings is flattened and then validated
    with schemas defined by Auto-FOX.
    Preset schemas are stored in :mod:`.schemas`.

    .. _schema: https://github.com/keleshev/schema

    Parameters
    ----------
    s : |plams.Settings|_
        A Settings instance containing all ARMC settings.

    Returns
    -------
    |plams.Settings|_:
        A validated Settings instance.

    """
    md_settings = s.job.pop('md_settings')
    preopt_settings = s.job.pop('preopt_settings')
    pes_settings = s.pop('pes')

    # Flatten the Settings instance
    s_flat = Settings()
    for k, v in s.items():
        try:
            s_flat[k] = v.flatten(flatten_list=False)
        except (AttributeError, TypeError):
            s_flat[k] = v

    # Validate the Settings instance
    s_flat.psf = schema_psf.validate(s_flat.psf)
    s_flat.job = schema_job.validate(s_flat.job)
    s_flat.hdf5_file = schema_hdf5.validate(s_flat.hdf5_file)
    s_flat.armc = schema_armc.validate(s_flat.armc)
    s_flat.move = schema_move.validate(s_flat.move)
    s_flat.param = schema_param.validate(s_flat.param)
    s_flat.molecule = validate_mol(s_flat.molecule)
    for k, v in pes_settings.items():
        schema_pes = get_pes_schema(k)
        pes_settings[k] = schema_pes.validate(v)

    # Unflatten and return
    s_ret = Settings()
    for k, v in s_flat.items():
        try:
            s_ret[k] = v.unflatten()
        except AttributeError:
            s_ret[k] = v

    s_ret.job.md_settings = md_settings
    s_ret.job.md_settings += get_template('md_cp2k_template.yaml')
    s_ret.job.preopt_settings = preopt_settings
    s_ret.pes = pes_settings
    return s_ret


def validate_mol(mol: Union[MultiMolecule, str, Iterable]) -> Tuple[MultiMolecule]:
    """Validate the ``"molecule"`` block and return a tuple of MultiMolecule instance(s).

    Parameters
    ----------
    mol : |str|_, |FOX.MultiMolecule|_ or |tuple|_ [|str|_, |FOX.MultiMolecule|_]
        The path+filename of one or more .xyz file(s) or :class:`.MultiMolecule` instance(s).
        Multiple instances can be supplied by wrapping them in an iterable (*e.g.* a :class:`tuple`)

    Returns
    -------
    |tuple|_ [|FOX.MultiMolecule|_]
        A tuple consisting of one or more :class:`.MultiMolecule` instances.

    Raises
    ------
    TypeError
        Raised if **mol** (or one of its elements) is of an invalid object type.

    """
    err = ("molecule expects one or more FOX.MultiMolecule instance(s) or .xyz filename(s); "
           "observed type: '{}'")

    def _validate(item: Union[MultiMolecule, str, abc.Iterable]) -> Optional[tuple]:
        """Validate the object type of **item**."""
        if isinstance(item, MultiMolecule):
            item.round(3)
            return (item,)
        elif isinstance(item, str):
            ret = MultiMolecule.from_xyz(item)
            ret.round(3)
            return (ret,)
        elif not isinstance(item, abc.Iterable):
            raise TypeError(err.format(item.__class__.__name__))
        return None

    # Validate **mol**
    ret = _validate(mol)
    if ret is not None:
        return ret

    # **mol** is an iterable, validate its elements
    ret = ()
    for i in mol:
        ret += _validate(i)

    # The to-be returned value is somehow an empty tuple; raise a TypeError
    if not ret:
        raise TypeError(err.format(None))
    return ret



def _parse_psf(s: Settings, path: str) -> Optional[PSFContainer]:
    s.md_settings = [s.md_settings.copy() for _ in s.molecule]
    psf = [_generate_psf(s, path, i) for i, _ in enumerate(s.molecule)]
    del s.psf

    if all(i is None for i in psf):
        return None

    for md_settings, i in zip(s.md_settings, psf):
        set_subsys_kind(md_settings, i.atoms)
    return psf


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


def _generate_psf(s: Settings, path: str, i: int) -> Optional[PSFContainer]:
    mol = s.molecule[i]
    md_settings = s.md_settings[i]
    param = s.param
    psf_s = s.psf

    if psf_s.psf_file:
        psf_file = psf_s.psf_file if isinstance(psf_s.psf_file, str) else psf_s.psf_file[i]
        return _read_psf(psf_file, param, md_settings)

    not_None = (psf_s.str_file or psf_s.rtf_file) and psf_s.ligand_atoms
    if not_None:
        atom_subset = set(mol.atoms).intersection(psf_s.ligand_atoms)
        mol.guess_bonds(atom_subset=list(atom_subset))

    # Create a and sanitize a plams molecule
    plams_mol = mol.as_Molecule(0)[0]
    res_list = mol.residue_argsort(concatenate=False)
    _assign_residues(plams_mol, res_list)

    # Initialize and populate the psf instance
    psf = PSFContainer()
    psf.generate_bonds(plams_mol)
    psf.generate_angles(plams_mol)
    psf.generate_dihedrals(plams_mol)
    psf.generate_impropers(plams_mol)
    psf.generate_atoms(plams_mol)
    psf.charge = 0.0

    # Overlay the PSFContainer instance with either the .rtf or .str file
    if not_None:
        psf.filename = join(path, f'mol.{i}.psf')
        str_, rtf = psf_s.str_file, psf_s.rtf_file
        if str_:
            overlay_str_file(psf, (str_ if isinstance(str_, str) else str_[i]))
        else:
            overlay_rtf_file(psf, (rtf if isinstance(rtf, str) else rtf[i]))
        md_settings.input.force_eval.subsys.topology.conn_file_name = psf.filename
        md_settings.input.force_eval.subsys.topology.conn_file_format = 'PSF'

    # Calculate the number of pairs
    param['count'].update(get_atom_count(param.index, psf.atom_type))
    return psf if not_None else None


def _read_psf(psf_file: str, param: pd.DataFrame, s: Settings) -> PSFContainer:
    psf = PSFContainer.read(psf_file)

    # Update the CP2K Settings
    s.input.force_eval.subsys.topology.conn_file_name = psf.filename
    s.input.force_eval.subsys.topology.conn_file_format = 'PSF'

    # Calculate the number of pairs
    param['count'].update(get_atom_count(param.index, psf.atom_type))
    return psf


def _assign_residues(plams_mol: Molecule, res_list: Iterable[Iterable[int]]) -> None:
    fix_bond_orders(plams_mol)
    res_name = 'COR'
    for i, j_list in enumerate(res_list, 1):
        for j in j_list:
            j += 1
            plams_mol[j].properties.pdb_info.ResidueNumber = i
            plams_mol[j].properties.pdb_info.ResidueName = res_name
        res_name = 'LIG'
