"""A module for parsing and sanitizing :class:`FOX.classes.monte_carlo.ARMC` settings."""

from typing import (Callable, Tuple, Any)
from os import getcwd
from os.path import isfile, isdir, split, join

import numpy as np
import pandas as pd

from scm.plams import Settings
from scm.plams.core.basejob import Job
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from .utils import (get_template, _get_move_range, dict_to_pandas)
from .cp2k_utils import set_keys
from .charge_utils import get_charge_constraints
from ..classes.psf_dict import PSFDict
from ..classes.multi_mol import MultiMolecule

__all__ = ['init_armc_sanitization']


TYPE_ERR = "{} expects an object of type '{}', not '{}'"
TYPE_ERR2 = "{} expects an object of type '{}' or '{}', not '{}'"
TYPE_DICT = {
    'rdf': MultiMolecule.init_rdf,
    'adf': MultiMolecule.init_adf,
    'rmsd': MultiMolecule.init_rmsd,
    'rmsf': MultiMolecule.init_rmsf,
    'init_rdf': MultiMolecule.init_rdf,
    'init_adf': MultiMolecule.init_adf,
    'init_rmsd': MultiMolecule.init_rmsd,
    'init_rmsf': MultiMolecule.init_rmsf,
}


def init_armc_sanitization(dict_: dict) -> Tuple[MultiMolecule, pd.DataFrame, Settings]:
    """Initialize the armc input settings sanitization."""
    s = Settings(dict_)

    s.job, mol = sanitize_job(s.job, s.molecule, s.param.prm_file)
    s.job.psf = generate_psf(mol, s.psf, s.param, s.job)
    del s.param.prm_file
    del s.molecule
    del s.psf

    s.param = sanitize_param(s.param, s.job.settings)
    s.pes = sanitize_pes(s.pes, mol)
    s.hdf5_file = sanitize_hdf5_file(s.hdf5_file)
    s.move = sanitize_move(s.move)
    s.armc, s.phi = sanitize_armc(s.armc)

    param = s.pop('param')
    param['param'] = param['param'].astype(float, copy=False)

    return mol, param, s


def finalize_settings(s: Settings) -> Settings:
    """Post-processing of **settings**."""
    if not s.psf:
        s.psf = False


def generate_psf(mol: MultiMolecule,
                 psf: Settings,
                 param: Settings,
                 job: Settings) -> Settings:
    """Generate the job.psf block."""
    psf_file = join(job.path, 'mol.psf')

    if psf:
        mol.guess_bonds(atom_subset=psf.ligand_atoms)
        psf_dict: PSFDict = PSFDict.from_multi_mol(mol)
        psf_dict.filename = psf_file
        psf_dict.update_atom_type(psf.str_file)
        job.settings.input.force_eval.subsys.topology.conn_file_name = psf_file
        job.settings.input.force_eval.subsys.topology.conn_file_format = 'PSF'
    else:
        psf_dict: PSFDict = PSFDict.from_multi_mol(mol)
        psf_dict.filename = np.array([False])

    for at, charge in param.charge.items():
        assert_type(at, str, 'param.')
        assert_type(charge, (float, np.float), 'param.charge.' + at)
        psf_dict.update_atom_charge(at, charge)

    return psf_dict


def get_name(item: Any) -> str:
    """Return the class name of **item**."""
    return item.__class__.__name__


def assert_type(item: Any,
                item_type: Callable,
                name: str = 'argument') -> None:
    if isinstance(item_type, tuple):
        type1 = item_type[0].__name__
    else:
        type1 = item_type.__name__

    if not isinstance(item, item_type):
        raise TypeError(TYPE_ERR.format(name, type1, get_name(item)))


def sanitize_armc(armc: Settings) -> Tuple[Settings, Settings]:
    """Sanitize the armc block."""
    assert_type(armc.iter_len, (int, np.integer), 'armc.iter_len')
    assert_type(armc.sub_iter_len, (int, np.integer), 'armc.sub_iter_len')
    if armc.sub_iter_len > armc.iter_len:
        raise ValueError('armc.sub_iter_len is larger than armc.iter_len')

    if hasattr(armc.gamma, '__index__'):
        armc.gamma = float(armc.gamma)
    assert_type(armc.gamma, (float, np.float), 'armc.gamma')

    if hasattr(armc.a_target, '__index__'):
        armc.a_target = float(armc.a_target)
    assert_type(armc.a_target, (float, np.float), 'armc.a_target')

    if hasattr(armc.phi, '__index__'):
        armc.phi = float(armc.phi)
    assert_type(armc.phi, (float, np.float), 'armc.phi')

    phi = Settings()
    phi.phi = armc.pop('phi')
    phi.kwarg = {}
    phi.func = np.add

    return armc, phi


def sanitize_param(param: Settings,
                   settings: Settings) -> Settings:
    """Sanitize the param block."""
    def check_key1_type(key1):
        if not isinstance(key1, str):
            error = TYPE_ERR.format('param.{}'.format(str(key1)) + ' key', 'str', get_name(key1))
            raise TypeError(error)
        elif not isinstance(value1, dict):
            error = TYPE_ERR.format('param.{}'.format(str(key1)) + ' value',
                                    'dict', get_name(value1))
            raise TypeError(error)

    def check_key2_type(key2):
        if not isinstance(key2, str):
            error = TYPE_ERR.format('param.{}.{}'.format(str(key1), str(key2)) +
                                    ' key', 'str', get_name(key2))
            raise TypeError(error)
        elif isinstance(value2, (int, np.int)):
            param[key1][key2] = float(value2)
        elif not isinstance(value2, (float, np.float, str)):
            error = TYPE_ERR.format('param.{}.{}'.format(str(key1), str(key2)),
                                    'float', get_name(value2))
            raise TypeError(error)

    for key1, value1 in param.items():
        check_key1_type(key1)
        for key2, value2 in value1.items():
            check_key2_type(key2)

    param = dict_to_pandas(param.as_dict(), 'param')
    param['key'] = set_keys(settings, param)
    param['param_old'] = np.nan
    return param


def sanitize_pes(pes: Settings,
                 mol: MultiMolecule) -> Settings:
    """Sanitize the pes block."""
    def check_key_type(key, value):
        assert_type(key, str)
        if isinstance(value.func, str):
            try:
                value.func = TYPE_DICT[value.func.lower().split('.')[-1]]
            except KeyError:
                raise KeyError("No type conversion available for '{}', consider directly passing"
                               " '{}' as type object".format(*[value.func.__name__]*2))
        assert_type(value.kwarg, dict, 'pes'+str(key)+'kwarg')

    for key, value in pes.items():
        for i in value.kwarg.values():
            try:
                i.sort()
            except AttributeError:
                pass
        check_key_type(key, value)
        value.ref = value.func(mol, **value.kwarg)
    return pes


def sanitize_hdf5_file(hdf5_file: str) -> str:
    """Sanitize the hdf5_file block."""
    if not isinstance(hdf5_file, str):
        raise TypeError(TYPE_ERR.format('hdf5_file', 'str', get_name(hdf5_file)))
    return hdf5_file


def sanitize_job(job: Settings,
                 mol: str,
                 prm_file: str) -> Settings:
    """Sanitize the job block."""
    if isinstance(mol, MultiMolecule):
        pass
    elif isinstance(mol, str):
        mol = MultiMolecule.from_xyz(mol)
    else:
        raise TypeError(TYPE_ERR2.format('job.molecule', 'FOX.MultiMolecule',
                                         'str', get_name(job.molecule)))

    if job.func.lower() == 'cp2kjob':
        job.func = Cp2kJob
    elif not isinstance(job.func, Job):
        raise TypeError(TYPE_ERR.format('job.func', 'Job', get_name(job.func)))

    if isinstance(job.settings, str):
        if isfile(job.settings):
            head, tail = split(job.settings)
            job.settings = get_template(tail, path=head)
        else:
            job.settings = get_template(job.settings)
    elif not isinstance(job.settings, dict):
        raise TypeError(TYPE_ERR2.format('job.settings', 'dict', 'str', get_name(job.settings)))
    job.settings.soft_update(get_template('md_cp2k_template.yaml'))

    assert_type(job.path, str, 'job.path')
    if job.path.lower() in ('.', 'pwd', 'cwd', '$pwd'):
        job.path = getcwd()
    assert isdir(job.path)

    assert_type(job.name, str, 'job.name')
    assert_type(job.folder, str, 'job.workdir')
    assert_type(job.keep_files, bool, 'job.keep_files')
    if prm_file:
        assert_type(prm_file, str, 'param.prm_file')
        job.settings.input.force_eval.mm.forcefield.parm_file_name = join(job.path, prm_file)
        job.settings.input.force_eval.mm.forcefield.parmtype = 'CHM'
    else:
        job.settings.input.force_eval.mm.forcefield.parmtype = 'OFF'
    job.settings.input['global'].project = job.name

    return job, mol


def sanitize_move(move: Settings) -> Settings:
    """Sanitize the move block."""
    move.range = _get_move_range(**move.range)
    move.func = np.multiply
    move.kwarg = {}
    if move.charge_constraints:
        move.charge_constraints = get_charge_constraints(move.charge_constraints)
    return move
