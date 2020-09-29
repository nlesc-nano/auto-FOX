"""A module for parsing and sanitizing ARMC settings.

Index
-----
.. currentmodule:: FOX.armc.sanitization
.. autosummary::
    dict_to_armc

API
---
.. autofunction:: dict_to_armc

"""

import os
import copy
from pathlib import Path
from itertools import islice
from collections import abc, Counter
from typing import (
    Union, Iterable, Tuple, Optional, Mapping, Any, MutableMapping, Hashable,
    Dict, TYPE_CHECKING, Generator, List, Collection, TypeVar, overload, cast
)

import numpy as np
import pandas as pd

from scm.plams import Molecule
from nanoutils import Literal, TypedDict, split_dict

from .guess import guess_param
from .mc_post_process import AtomsFromPSF
from .schemas import (
    validate_phi, validate_pes, validate_monte_carlo, validate_psf,
    validate_job, validate_sub_job, validate_param, validate_main,
    PESDict, PESMapping, PhiMapping, MainMapping, ParamMapping_, MCMapping,
    PSFMapping, JobMapping
)

from ..utils import get_atom_count
from ..io.read_psf import PSFContainer, overlay_str_file, overlay_rtf_file
from ..classes import MultiMolecule
from ..functions.cp2k_utils import UNIT_MAP
from ..functions.molecule_utils import fix_bond_orders, residue_argsort
from ..functions.charge_parser import assign_constraints

if TYPE_CHECKING:
    from .package_manager import PackageManager, PkgDict
    from .param_mapping import ParamMapping
    from .phi_updater import PhiUpdater
    from .monte_carlo import MonteCarloABC
else:
    from ..type_alias import PackageManager, ParamMapping, PhiUpdater, MonteCarloABC
    from builtins import dict as PkgDict

__all__ = ['dict_to_armc']

T = TypeVar('T')
KT = TypeVar('KT', bound=Hashable)
MT = TypeVar('MT', bound=Mapping[Any, Any])


class RunDict(TypedDict, total=False):
    """A :class:`~typing.TypedDict` representing the input of :func:`run_armc`."""

    path: Union[str, os.PathLike]
    folder: Union[str, os.PathLike]
    logfile: Union[str, os.PathLike]
    psf: Optional[List[PSFContainer]]


def dict_to_armc(input_dict: MainMapping) -> Tuple[MonteCarloABC, RunDict]:
    """Initialize the armc input settings sanitization.

    Parameters
    ----------
    input_dict : :class:`dict`
        A dictionary containing all ARMC settings.

    Returns
    -------
    :class:`ARMC` and :class:`dict`
        An ARMC instance and a dictionary with keyword arguments for :func:`run_armc`.

    """
    dct = validate_main(copy.deepcopy(input_dict))

    # Construct an ARMC instance
    phi = get_phi(dct['phi'])
    package, mol_list = get_package(dct['job'], phi.phi)
    param, _param, _param_frozen = get_param(dct['param'])
    mc, run_kwargs = get_armc(dct['monte_carlo'], package, param, phi, mol_list)

    # Update the job Settings
    if _param_frozen is not None:
        package.update_settings(list(prm_iter(_param_frozen)), new_keys=True)
    package.update_settings(list(prm_iter(_param)), new_keys=True)

    # Handle psf stuff
    psf_list: Optional[List[PSFContainer]] = get_psf(dct['psf'], mol_list)
    run_kwargs['psf'] = psf_list
    update_count(param, psf=psf_list, mol=mol_list)
    _parse_ligand_alias(psf_list, prm=param)
    if psf_list is not None:
        mc.pes_post_process = [AtomsFromPSF.from_psf(*psf_list)]
        workdir = Path(run_kwargs['path']) / run_kwargs['folder']
        _update_psf_settings(package.values(), phi.phi, workdir)

    # Guess parameters
    if _param_frozen is not None:
        _guess_param(mc, _param_frozen, frozen=True, psf=psf_list)
    _guess_param(mc, _param, frozen=False, psf=psf_list)

    # Add PES evaluators
    pes = get_pes(dct['pes'])
    for name, kwargs in pes.items():
        mc.add_pes_evaluator(name, **kwargs)
    return mc, run_kwargs


def _guess_param(mc: MonteCarloABC, prm: dict,
                 frozen: bool = False,
                 psf: Optional[Iterable[PSFContainer]] = None) -> None:
    package = mc.package_manager
    settings = next(iter(package.values()))[0]['settings']
    prm_file = settings.get('prm')

    # Guess and collect all to-be updated parameters
    seq = []
    for k, v in prm_iter(prm):
        mode = v.pop('guess', None)
        if mode is None:
            continue
        param = v['param']
        unit = UNIT_MAP[v.get('unit', 'k_e' if param == 'epsilon' else 'angstrom')]

        prm_series = guess_param(mc.molecule, param, mode=mode,
                                 psf_list=psf, prm=prm_file, unit=unit)
        prm_dict = {' '.join(_k for _k in sorted(k)): v for k, v in prm_series.items()}
        prm_dict['param'] = param
        seq.append((k, prm_dict))

    # Update the constant parameters
    package.update_settings(seq, new_keys=True)
    if frozen:
        return

    # Update the variable parameters
    param_mapping = mc.param
    for k, v in seq:
        iterator = (((k, v['param'], at), value) for at, value in v.items() if at == 'param')
        for key, value in iterator:
            param_mapping['param'].loc[key] = value
            param_mapping['param_old'].loc[key] = value
            param_mapping['min'][key] = -np.inf
            param_mapping['max'][key] = np.inf
            param_mapping['constraints'][key] = None
            param_mapping['count'][key] = 0
    return


def get_phi(dct: PhiMapping) -> PhiUpdater:
    """Construct a :class:`PhiUpdater` instance from **dct**.

    Returns
    -------
    :class:`PhiUpdater`
        A PhiUpdater for :class:`~FOX.armc.ARMC`.

    """
    phi_dict = validate_phi(dct)
    phi_type = phi_dict.pop('type')  # type: ignore
    kwargs = phi_dict.pop('kwargs')  # type: ignore
    return phi_type(**phi_dict, **kwargs)


def get_package(dct: JobMapping, phi: Iterable) -> Tuple[PackageManager, Tuple[MultiMolecule, ...]]:
    """Construct a :class:`PackageManager` instance from **dct**.

    Returns
    -------
    :class:`~FOX.armc.PackageManager` and :class:`tuple` [:class:`~FOX.MultiMolecule`]
        A PackageManager and a tuple of MultiMolecules.

    """
    _sub_pkg_dict: Dict[str, Any] = split_dict(
        dct, preserve_order=True, keep_keys={'type', 'molecule'}
    )

    job_dict = validate_job(dct)
    mol_list = [mol.as_Molecule(mol_subset=0)[0] for mol in job_dict['molecule']]

    data: Dict[str, List[PkgDict]] = {}
    for k, v in _sub_pkg_dict.items():
        data[k] = []
        for _ in phi:
            for mol in mol_list:
                kwargs = validate_sub_job(v)
                kwargs['molecule'] = mol.copy()

                pkg_name = kwargs['type'].pkg_name
                kwargs['settings'].specific[pkg_name].soft_update(kwargs.pop('template'))
                data[k].append(kwargs)

    pkg_type = job_dict['type']
    return pkg_type(data), job_dict['molecule']


def get_param(dct: ParamMapping_) -> Tuple[ParamMapping, dict, dict]:
    """Construct a :class:`ParamMapping` instance from **dct**.

    Returns
    -------
    :class:`ParamMapping`, :class:`dict`, :class:`dict`
        A ParamMapping, a parameter dictionary and
        a parameter dictionary with all frozen parameters.

    """
    _prm_dict = dct
    _sub_prm_dict = split_dict(
        _prm_dict, preserve_order=True, keep_keys={'type', 'move_range', 'func', 'kwargs'}
    )
    _sub_prm_dict_frozen = _get_prm_frozen(_sub_prm_dict)

    prm_dict = validate_param(_prm_dict)
    kwargs = prm_dict.pop('kwargs')
    data = _get_param_df(_sub_prm_dict)
    constraints, min_max = _get_prm_constraints(_sub_prm_dict)
    data[['min', 'max']] = min_max

    for *_key, value in _get_prm(_sub_prm_dict_frozen):
        key = tuple(_key)
        data.loc[key, :] = [value, True, -np.inf, np.inf]
    data.sort_index(inplace=True)

    param_type = prm_dict.pop('type')  # type: ignore
    return (
        param_type(data, constraints=constraints, **prm_dict, **kwargs),
        _sub_prm_dict,
        _sub_prm_dict_frozen,
    )


def get_pes(dct: Mapping[str, PESMapping]) -> Dict[str, PESDict]:
    """Construct a :class:`dict` with PES-descriptor workflows."""
    return {k: validate_pes(v) for k, v in dct.items()}


def get_armc(dct: MCMapping,
             package_manager: PackageManager,
             param: ParamMapping,
             phi: PhiUpdater,
             mol: Iterable[MultiMolecule]) -> Tuple[MonteCarloABC, RunDict]:
    """Construct an :class:`ARMC` instance from **dct**.

    Returns
    -------
    :class:`~FOX.armc.ARMC` and :class:`dict`
        A ARMC instance and a dictionary with keyword arguments for :func:`FOX.armc.run_armc`.

    """
    mc_dict = validate_monte_carlo(dct)

    pop_keys = ('path', 'folder', 'logfile')
    kwargs = {k: mc_dict.pop(k) for k in pop_keys}  # type: ignore

    workdir = os.path.join(kwargs['path'], kwargs['folder'])
    logfile = kwargs['logfile']
    hdf5 = mc_dict['hdf5_file']
    if not os.path.isdir(os.path.dirname(logfile)):
        kwargs['logfile'] = os.path.join(workdir, logfile)
    else:
        kwargs['logfile'] = os.path.abspath(logfile)

    if not os.path.isdir(os.path.dirname(hdf5)):
        mc_dict['hdf5_file'] = os.path.join(workdir, hdf5)
    else:
        kwargs['hdf5_file'] = os.path.abspath(hdf5)

    mc_type = mc_dict.pop('type')  # type: ignore
    return mc_type(phi=phi, param=param, package_manager=package_manager,
                   molecule=mol, **mc_dict), cast(RunDict, kwargs)


def get_psf(dct: PSFMapping, mol_list: Iterable[MultiMolecule]
            ) -> Optional[List[PSFContainer]]:
    """Construct a list of :class:`PSFContainer` instances from **dct**.

    Returns
    -------
    :class:`list` [:class:`~FOX.PSFContainer`], optional
        If either the ``"psf_file"``, ``"rtf_file"`` or ``"str_file"`` key is present in **dct**
        then return a list of PSFContaisers.
        Return ``None`` otherwise.

    """
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
    """Construct a list of :class:`~FOX.PSFContainer` instances."""
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


def _psf_idx_iterator(job_list: Iterable[T], phi: Iterable) -> Generator[Tuple[int, T], None, None]:
    job_iterator = iter(job_list)
    for i, job in enumerate(job_iterator):
        yield i, job

        phi_iterator = islice(phi, 1)
        for _, job in zip(phi_iterator, job_iterator):
            yield i, job


def _update_psf_settings(job_lists: Iterable[Iterable[MutableMapping[str, Any]]], phi: Iterable,
                         workdir: Union[str, os.PathLike]) -> None:
    """Set the .psf path in all job settings."""
    for job_list in job_lists:
        iterator = _psf_idx_iterator(job_list, phi)
        for i, job in iterator:
            job['settings'].psf = os.path.join(workdir, f'mol.{i}.psf')


PrmTuple = Tuple[str, str, str, float]


def prm_iter(dct: Mapping[KT, Union[MT, Iterable[MT]]]
             ) -> Generator[Tuple[KT, MT], None, None]:
    """Create a an iterator yielding individual parameter dictionaries.

    Yields
    ------
    :class:`~collections.abc.Hashable` and :class:`~collections.abc.Mapping`
        An iterator yielding the super-keys of **dct** and its nested dictionaries.

    """
    for key_alias, _dct_list in dct.items():

        # Ensure that we're dealing with a list of dicts
        if isinstance(_dct_list, abc.Mapping):
            dct_list: Iterable[MT] = [_dct_list]
        else:
            dct_list = _dct_list

        for sub_dict in dct_list:
            yield (key_alias, sub_dict)


def _get_param_df(dct: Mapping[str, Any]) -> pd.DataFrame:
    """Construct a DataFrame for :class:`ParamMapping`.

    Returns
    -------
    :class:`pandas.DataFrame`
        A parameter DataFrame with a three-level :class:`~pandas.MultiIndex` as index.
        The available columns are: ``"param"``, ``"constraints"``, ``"min"`` and ``"max"``.

    """
    columns = ['key', 'param_type', 'atoms', 'param']
    data = _get_prm(dct)

    df = pd.DataFrame(data, columns=columns)
    df.set_index(['key', 'param_type', 'atoms'], inplace=True)

    df['constant'] = False
    df['min'] = -np.inf
    df['max'] = np.inf
    return df


def _get_prm(dct: Mapping[str, Union[Mapping, Iterable[Mapping]]]
             ) -> Generator[PrmTuple, None, None]:
    """Create a generator yielding DataFrame rows for :func:`_get_param_df`.

    Yields
    ------
    :class:`str`, :class:`str`, :class:`str` and :class:`float`
        A generator yielding 4-tuples.
        The first three elements represent :class:`~pandas.MultiIndex` keys while the
        last one is the actual parameter value.

    """
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
    """Return a parameter dictionary for all frozen parameters.

    Returns
    -------
    :class:`dict` [:class:`str`, :class:`list` [:class:`dict`]], optional
        If not ``None``, a dictionary consting of lists of dictionaries.
        The list-embedded dictionary consist of normal parameter dictionaries,
        except that it represents constant (rather than variable) parameters.

    """
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


def _get_prm_constraints(
    dct: Mapping[str, Union[MutableMapping, Iterable[MutableMapping]]]
) -> Tuple[
    Dict[Tuple[str, str], Optional[List[Dict[str, float]]]],
    List[Tuple[float, float]]
]:
    """Parse all user-provided constraints.

    Yields
    ------
    :class:`dct` (optional), :class:`float` and :class:`float`
        A generator yielding a tuple of three values for each parameter:

        1. A dictionary of constraints (optional).
        2. The parameter's minimum value.
        3. The parameter's maximum value.

    """
    ignore_keys = {'frozen', 'constraints', 'param', 'unit', 'guess'}

    constraints_dict: Dict[Tuple[str, str], Optional[List[Dict[str, float]]]] = {}
    min_max: List[Tuple[float, float]] = []

    dct_iterator = prm_iter(dct)
    for key_alias, sub_dict in dct_iterator:
        try:
            proto_constraints = sub_dict.pop('constraints')
        except KeyError:
            for k in sub_dict:
                if k not in ignore_keys:
                    min_max.append((-np.inf, np.inf))
            constraints_dict[key_alias, sub_dict["param"]] = []
            continue

        extremites, constraints = assign_constraints(proto_constraints)
        for k in sub_dict:
            if k not in ignore_keys:
                min_max.append((
                    extremites.get((k, 'min'), -np.inf),
                    extremites.get((k, 'max'), np.inf)
                ))
        constraints_dict[key_alias, sub_dict["param"]] = (
            constraints if constraints is not None else []
        )
    return constraints_dict, min_max


def _parse_ligand_alias(psf_list: Optional[List[PSFContainer]], prm: ParamMapping) -> None:
    """Replace ``$LIGAND`` constraints with explicit ligands."""
    not_implemented = psf_list is None or len(psf_list) != 1
    if not not_implemented:
        psf: PSFContainer = psf_list[0]
        lig_id = psf.residue_id.iloc[-1]
        df = psf.atoms[psf.residue_id == lig_id]

        atom_types: Iterable[str] = df['atom type'].values
        atom_counter = Counter(atom_types)
        update = False
        for k in atom_counter:
            key = ('charge', 'charge', k)
            if key not in prm['param'].index:
                update = True
                prm['param'].loc[key] = df.loc[df['atom type'] == k, 'charge'].iloc[0]
                prm['min'].at[key] = -np.inf
                prm['max'].at[key] = np.inf
                prm['constant'].at[key] = True

        if update:
            prm['param'].sort_index(inplace=True)
            prm['min'].sort_index(inplace=True)
            prm['max'].sort_index(inplace=True)
            prm['constant'].sort_index(inplace=True)

    for lst in prm.constraints.values():
        for series in lst:
            if "$LIGAND" not in series:
                continue
            elif not_implemented:
                raise NotImplementedError

            i = series.pop("$LIGAND")
            for k, v in atom_counter.items():
                if k in series:
                    series[k] += v * i
                else:
                    series[k] = v * i


@overload
def update_count(param: ParamMapping, psf: Iterable[PSFContainer], mol: None) -> None: ...
@overload
def update_count(param: ParamMapping, psf: None, mol: Iterable[MultiMolecule]) -> None: ...
def update_count(param, psf=None, mol=None):  # noqa: E302
    """Assign atom-counts to the passed :class:`ParamMapping`."""
    # Construct a generator
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
