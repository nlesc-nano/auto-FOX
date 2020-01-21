"""
FOX.armc_functions.sanitization
===============================

A module for parsing and sanitizing ARMC settings.

"""

import os
import functools
from typing import Union, Iterable, Tuple, Optional, Mapping
from os.path import join, isfile, abspath
from collections import abc

import numpy as np
import pandas as pd

from scm.plams import Settings, Molecule

from ..io.read_psf import PSFContainer, overlay_str_file, overlay_rtf_file
from ..classes.multi_mol import MultiMolecule
from ..functions.utils import get_template, dict_to_pandas, get_atom_count, _get_move_range
from ..functions.cp2k_utils import set_keys, set_subsys_kind
from ..functions.molecule_utils import fix_bond_orders
from ..armc_functions.schemas import (
    get_pes_schema, schema_armc, schema_move, schema_job, schema_param, schema_hdf5, schema_psf
)

__all__ = ['init_armc_sanitization']


def init_armc_sanitization(dct: Mapping) -> Settings:
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
    # Load and apply the template
    s_inp = get_template('armc_template.yaml')
    s_inp.update(dct)

    # Validate, post-process and return
    s = validate(s_inp)
    _parse_move(s)
    _parse_armc(s)
    job: Settings = _parse_job(s)
    pes: Settings = _parse_pes(s)

    _parse_param(s, job)
    job.psf = _parse_psf(s, job.path)
    _parse_preopt(s)
    return s, pes, job


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


def _parse_move(s: Settings) -> None:
    move = s.move
    s.apply_move = functools.partial(move.func, *move.args, **move.kwargs)
    s.move_range = _get_move_range(**move.range)
    del s.move


def _parse_job(s: Settings) -> Settings:
    job = s.job
    if job.path == '.' or not job.path:
        job.path = os.getcwd()

    s.job_type = functools.partial(job.pop('job_type'), name=job.pop('name'))
    s.md_settings = job.pop('md_settings')
    s.preopt_settings = job.pop('preopt_settings')
    s.keep_files = job.pop('keep_files')
    s.rmsd_threshold = job.pop('rmsd_threshold')
    return s.pop('job')


def _parse_armc(s: Settings) -> None:
    armc = s.armc

    s.a_target = armc.a_target
    s.gamma = armc.gamma
    s.iter_len = armc.iter_len
    s.sub_iter_len = armc.sub_iter_len
    s.phi = armc.phi
    s.apply_phi = np.add

    # Delete leftovers
    del s.armc


def _parse_pes(s: Settings) -> Settings:
    return s.pop('pes')


def _parse_psf(s: Settings, path: str) -> Optional[PSFContainer]:
    s.md_settings = [s.md_settings.copy() for _ in s.molecule]
    psf = [_generate_psf(s, path, i) for i, _ in enumerate(s.molecule)]
    del s.psf

    if all(i is None for i in psf):
        return None

    for md_settings, i in zip(s.md_settings, psf):
        set_subsys_kind(md_settings, i.atoms)
    return psf


def _parse_preopt(s: Settings) -> None:
    if s.preopt_settings is None:
        return

    if s.preopt_settings is True:
        s.preopt_settings = [Settings() for _ in s.md_settings]

    for md, preopt in zip(s.md_settings, s.preopt_settings):
        preopt += md
        del preopt.input.motion.md
        preopt.input['global'].run_type = 'geometry_optimization'


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
