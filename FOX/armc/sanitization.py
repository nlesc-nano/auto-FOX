"""A module for parsing and sanitizing ARMC settings.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    dict_to_armc

API
---
.. autofunction:: dict_to_armc

"""

from __future__ import annotations

import os
import math
import copy
import warnings
from pathlib import Path
from itertools import chain, repeat
from collections import abc, Counter
from typing import (
    Union, Iterable, Tuple, Optional, Mapping, Any, MutableMapping, Iterator,
    Dict, TYPE_CHECKING, Generator, List, Collection, TypeVar, overload, cast,
    Sequence,
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
    PSFMapping, JobMapping, ValidationDict, validation_schema, SubJobMapping
)

from ..utils import get_atom_count
from ..io.read_psf import PSFContainer, overlay_str_file, overlay_rtf_file
from ..classes import MultiMolecule
from ..functions.cp2k_utils import UNIT_MAP
from ..functions.molecule_utils import fix_bond_orders, residue_argsort
from ..functions.charge_parser import assign_constraints
from ..functions.sorting import sort_param

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
KT = TypeVar('KT')
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
    param, _param, _param_frozen, validation_dict = get_param(dct['param'])
    mc, run_kwargs = get_armc(dct['monte_carlo'], package, param, phi, mol_list)

    # Update the job Settings
    if _param_frozen is not None:
        package.update_settings(list(prm_iter(_param_frozen)), new_keys=True)
    package.update_settings(list(prm_iter(_param)), new_keys=True)

    # Handle psf stuff
    psf_list: Optional[List[PSFContainer]] = get_psf(dct['psf'], mol_list)
    run_kwargs['psf'] = psf_list
    _parse_ligand_alias(psf_list, prm=param)
    if psf_list is not None:
        mc.pes_post_process = [AtomsFromPSF.from_psf(*psf_list)]  # type: ignore[assignment]
        workdir = Path(run_kwargs['path']) / run_kwargs['folder']
        _update_psf_settings(package.values(), phi.phi, workdir)  # type: ignore[arg-type]
        _update_psf_charge(param, psf_list)

    # Guess parameters
    if _param_frozen is not None:
        _guess_param(mc, _param_frozen, frozen=True, psf=psf_list)
    _guess_param(mc, _param, frozen=False, psf=psf_list)
    update_count(param, psf=psf_list, mol=mol_list)

    # Perform validation checks
    validate_atoms(param, psf_list, mol_list,
                   allow_non_existent=validation_dict['allow_non_existent'])
    param._set_net_charge()
    validate_charge(param._net_charge, tolerance=validation_dict['charge_tolerance'])
    validate_constraints(param, enforce_constraints=validation_dict['enforce_constraints'])
    param._net_charge_to_integer()

    mc.param.param.sort_index(inplace=True)
    mc.param.param_old.sort_index(inplace=True)
    mc.param.metadata.sort_index(inplace=True)

    # Add PES evaluators
    pes = get_pes(dct['pes'], len(mol_list))
    for name, kwargs in pes.items():
        mc.add_pes_evaluator(name, **kwargs)  # type: ignore[call-overload]

    pes2 = get_pes(dct['pes_validation'], len(mol_list))
    for name, kwargs in pes2.items():
        mc.add_pes_evaluator(name, validation=True, **kwargs)  # type: ignore[call-overload]
    return mc, run_kwargs


def validate_atoms(
    param: ParamMapping,
    psf_list: Optional[Iterable[PSFContainer]],
    mol_list: Iterable[MultiMolecule],
    allow_non_existent: bool = False,
) -> None:
    """Check if all atom types specified in **param** are actually present.

    If not :data:`None` use the psf containers for cross referencing;
    use the molecule otherwise.

    """
    iterator1 = chain.from_iterable(at.split() for at in param.param.index.levels[2])
    atoms = dict.fromkeys(iterator1, False)

    if psf_list is None:
        iterator2 = (at for mol in mol_list for at in mol.atoms)
    else:
        iterator2 = (at for psf in psf_list for at in set(psf.atom_type))
    for key in iterator2:
        atoms[key] = True

    if all(atoms.values()):
        return None

    unknows = sorted(k for k, v in atoms.items() if not v)
    msg = f"Found {len(unknows)} explicitly specified parameters for non-existent atoms: {unknows}"
    if allow_non_existent:
        warnings.warn(msg, category=RuntimeWarning, stacklevel=3)
    else:
        raise RuntimeError(msg)


def validate_charge(charge: None | np.ndarray, tolerance: float = 0.01) -> None:
    """Check if the net **charge** is integer within a given **tolerance**."""
    if charge is None:
        return

    delta = np.abs(charge - charge.round())
    if (delta > tolerance).any():
        raise ValueError(f'Net charge {charge} not integer within a tolerance {tolerance}')


def validate_constraints(param: ParamMapping, tolerance: float = 0.01,
                         enforce_constraints: bool = False) -> None:
    """Check that all user-specified parameters satisfy the user-specified constraints."""
    msg = ""

    # Check minima and maxima
    min_ok = param.metadata[0, 'min'] <= param.param[0]
    max_ok = param.metadata[0, 'max'] >= param.param[0]
    if not min_ok.all():
        msg += f"Parameters smaller then specified minima:\n    {param.param.loc[~min_ok, 0]}\n"
    if not max_ok.all():
        msg += f"Parameters larger then specified maxima:\n    {param.param.loc[~max_ok, 0]}\n"

    # Check ratios
    str_series = param.constraints_to_str()
    for tup2, series_tup in param.constraints.items():
        if series_tup is None or not len(series_tup):
            continue
        param_: pd.Series = param.param.loc[tup2, 0]

        series_iter = iter(series_tup)
        series = next(series_iter)
        sum_ref = (param_.loc[series.index] * series).sum()
        for series in series_iter:
            sum_ = (param_.loc[series.index] * series).sum()
            if not np.allclose(sum_, sum_ref, atol=tolerance):
                msg += f"Parameters {tup2} failed to satisfy the ratio:\n    {str_series.loc[tup2]}\n"  # noqa: E501
                break

    if msg:
        if enforce_constraints:
            raise RuntimeError(msg)
        else:
            warnings.warn(msg, category=RuntimeWarning, stacklevel=3)


def _guess_param(mc: MonteCarloABC, prm: dict,
                 frozen: bool = False,
                 psf: Optional[Iterable[PSFContainer]] = None) -> None:
    package = mc.package_manager
    settings = next(iter(package.values()))[0]['settings']
    prm_file = settings.get('prm')

    # Guess and collect all to-be updated parameters
    seq = []
    unit_lst = []
    for k, v in prm_iter(prm):
        mode = v.pop('guess', None)
        if mode is None:
            continue
        param = v['param']
        unit = UNIT_MAP[v.get('unit') or ('k_e' if param == 'epsilon' else 'angstrom')]
        unit_lst.append(unit)

        try:
            param_mapping = mc.param.param.loc[(k, param), 0].copy()
            param_mapping.index = pd.MultiIndex.from_tuples(i.split() for i in param_mapping.index)
        except KeyError:
            param_mapping = None

        prm_series = guess_param(
            mc.molecule, param, mode,
            psf_list=psf, prm=prm_file, unit=unit, param_mapping=param_mapping,
        )
        prm_dict = {' '.join(_k for _k in sorted(k)): v for k, v in prm_series.items()}
        prm_dict['param'] = param
        seq.append((k, prm_dict))

    # Update the constant parameters
    package.update_settings(seq, new_keys=True)

    # Update the variable parameters
    metadata = {'min': -np.inf, 'max': np.inf, 'count': 0, 'guess': True}
    _prm = mc.param.param
    for (k, v), unit in zip(seq, unit_lst):
        iterator = (((k, v['param'], at), value) for at, value in v.items() if at != 'param')
        for key, value in iterator:
            if key not in _prm.index:
                mc.param.add_param(key, value, frozen=frozen, unit=unit, **metadata)
    return


def get_phi(dct: PhiMapping) -> PhiUpdater:
    """Construct a :class:`PhiUpdater` instance from **dct**.

    Returns
    -------
    :class:`PhiUpdater`
        A PhiUpdater for :class:`~FOX.armc.ARMC`.

    """
    phi_dict = validate_phi(dct)
    phi_type = phi_dict.pop('type')  # type: ignore[misc]
    kwargs = phi_dict.pop('kwargs')  # type: ignore[misc]
    return phi_type(**phi_dict, **kwargs)  # type: ignore[return-value]


def get_package(dct: JobMapping, phi: Iterable) -> Tuple[PackageManager, Tuple[MultiMolecule, ...]]:
    """Construct a :class:`PackageManager` instance from **dct**.

    Returns
    -------
    :class:`~FOX.armc.PackageManager` and :class:`tuple` [:class:`~FOX.MultiMolecule`]
        A PackageManager and a tuple of MultiMolecules.

    """
    _sub_pkg_dict: Dict[str, SubJobMapping] = split_dict(  # type: ignore[call-overload]
        dct, preserve_order=True, keep_keys={'type', 'molecule', 'lattice'}
    )

    job_dict = validate_job(dct)
    mol_list = [mol.as_Molecule(mol_subset=0)[0] for mol in job_dict['molecule']]
    if job_dict['lattice'] is not None:
        if len(job_dict['molecule']) != len(job_dict['lattice']):
            raise ValueError("`job.molecule` and `job.lattice` must be of equal length")
        for m, lat in zip(job_dict['molecule'], job_dict['lattice']):
            m.lattice = lat

    data: Dict[str, List[PkgDict]] = {}
    for k, v in _sub_pkg_dict.items():
        data[k] = []
        kwargs = validate_sub_job(v)

        s_list = kwargs['settings']
        if len(s_list) == 1:
            kwargs['settings'] = [s_list[0].copy() for _ in mol_list]
        else:
            assert len(s_list) == len(mol_list)

        for _ in phi:
            for i, mol in enumerate(mol_list):
                kwargs2 = kwargs.copy()
                kwargs2['molecule'] = mol.copy()  # type: ignore[misc]

                pkg_name = kwargs2['type'].pkg_name
                kwargs2['settings'] = s_list[i]
                kwargs2['settings'].specific[pkg_name].soft_update(kwargs2.pop('template'))  # type: ignore[misc] # noqa: E501
                data[k].append(kwargs2)  # type: ignore[arg-type]

    pkg_type = job_dict['type']
    return pkg_type(data), job_dict['molecule']  # type: ignore[return-value]


def get_param(dct: ParamMapping_) -> Tuple[ParamMapping, dict, dict, ValidationDict]:
    """Construct a :class:`ParamMapping` instance from **dct**.

    Returns
    -------
    :class:`ParamMapping`, :class:`dict`, :class:`dict`
        A ParamMapping, a parameter dictionary and
        a parameter dictionary with all frozen parameters.

    """
    _prm_dict = dct
    _sub_prm_dict = split_dict(  # type: ignore[call-overload]
        _prm_dict, preserve_order=True,
        keep_keys={'type', 'move_range', 'func', 'kwargs', 'validation'}
    )
    _sub_prm_dict_frozen = _get_prm_frozen(_sub_prm_dict)

    prm_dict = validate_param(_prm_dict)
    kwargs = prm_dict.pop('kwargs')  # type: ignore[misc]
    validation_dict = validation_schema.validate(prm_dict.pop('validation'))  # type: ignore[misc]

    # Construct and fill the parameter dataframe
    data = _get_param_df(_sub_prm_dict)
    constraints, min_max, units = _get_prm_constraints(_sub_prm_dict)
    data[['min', 'max']] = min_max
    data['unit'] = units

    if _sub_prm_dict_frozen is not None:
        data2 = _get_param_df(_sub_prm_dict_frozen)
        if len(data2) != 0:
            constraints2, min_max2, units2 = _get_prm_constraints(_sub_prm_dict_frozen)
            data2[['min', 'max']] = min_max2
            data2['unit'] = units2
            data2['frozen'] = True
            data = data.append(data2)
    data.sort_index(inplace=True)

    param_type = prm_dict.pop('type')  # type: ignore
    return (  # type: ignore[return-value]
        param_type(data, constraints=constraints, **prm_dict, **kwargs),
        _sub_prm_dict,
        _sub_prm_dict_frozen,
        validation_dict,
    )


def get_pes(dct: Mapping[str, PESMapping], mol_len: int) -> Dict[str, PESDict]:
    """Construct a :class:`dict` with PES-descriptor workflows."""
    ret = {k: validate_pes(v) for k, v in dct.items()}

    iterator = (len(v['ref']) for v in ret.values() if v['ref'] is not None)
    assert all(mol_len == i for i in iterator)
    return ret


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


def _update_psf_settings(job_lists: Iterable[Iterable[MutableMapping[str, Any]]], phi: Collection,
                         workdir: Union[str, os.PathLike]) -> None:
    """Set the .psf path in all job settings."""
    for job_list in job_lists:
        if len(phi) == 1:
            iterator: Iterable[Tuple[int, MutableMapping[str, Any]]] = enumerate(job_list)
        else:
            iterator = ((0, job) for job in job_list)

        for i, job in iterator:
            job['settings'].psf = os.path.join(workdir, f'mol.{i}.psf')


def _update_psf_charge(param: ParamMapping, psf_list: Iterable[PSFContainer]) -> None:
    """Add all (frozen) charges in the .psf files to the passed ``ParamMapping``."""
    metadata = {'min': -np.inf, 'max': np.inf, 'count': 0, 'frozen': True,
                'guess': False, 'unit': ''}

    for i, psf in enumerate(psf_list):
        # Map atom types to atom charges and
        # validate that each atom type only has a single unique charge
        charge_dict = {}
        for _k, v in zip(psf.atom_type, psf.charge):
            k = ("charge", "charge", _k)
            if not math.isclose(charge_dict.get(k, v), v):
                v_old = charge_dict[k]
                raise ValueError(f"Found multiple charges for atom type {_k!r}: {v_old} and {v}")
            charge_dict[k] = v

        # Update the charges in `param`
        for k, v in charge_dict.items():  # type: Tuple[str, str, str], float
            if k not in param.param.index:
                param.add_param(k, v, **metadata)


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
            dct_list: Iterable[MT] = [_dct_list]  # type: ignore[list-item]
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
    _sort_atoms(df)
    df.set_index(['key', 'param_type', 'atoms'], inplace=True)

    df['param_old'] = df['param'].copy()
    df['frozen'] = False
    df['guess'] = False
    df['min'] = -np.inf
    df['max'] = np.inf
    df['count'] = 0
    df['unit'] = ''
    return df


def _sort_atoms(df: pd.DataFrame) -> None:
    """Sort all 2- and 3-atom based parameters in alphabetical order.

    Only the first and last elements are sorted in the case of 3-atom based parameters.

    Parameters
    ----------
    df : :class:`pandas.DataFrame`
        A dataframe with the ``"param_type"`` and ``"atoms"`` columns.

    Raises
    ------
    KeyError : Raised if duplicate keys are observed after the sorting.

    """
    param_types = set(df["param_type"])
    for prm in param_types:
        condition = df['param_type'] == prm
        atoms = df.loc[condition, 'atoms'].values.astype(np.str_)
        try:
            df.loc[condition, 'atoms'] = sort_param(atoms)
        except NotImplementedError as ex:
            warning = RuntimeWarning(str(ex))
            warning.__cause__ = ex
            warnings.warn(warning)


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
    List[Tuple[float, float]],
    List[str],
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
    units: List[str] = []

    dct_iterator = prm_iter(dct)
    for key_alias, sub_dict in dct_iterator:
        unit = sub_dict.setdefault('unit', None) or ''

        try:
            proto_constraints = sub_dict.pop('constraints')
        except KeyError:
            for k in sub_dict:
                if k not in ignore_keys:
                    units.append(unit)
                    min_max.append((-np.inf, np.inf))
            constraints_dict[key_alias, sub_dict["param"]] = []
            continue

        extremites, constraints = assign_constraints(proto_constraints)
        for k in sub_dict:
            if k not in ignore_keys:
                units.append(unit)
                min_max.append((
                    extremites.get((k, 'min'), -np.inf),
                    extremites.get((k, 'max'), np.inf)
                ))
        constraints_dict[key_alias, sub_dict["param"]] = (
            constraints if constraints is not None else []
        )
    return constraints_dict, min_max, units


def _parse_ligand_alias(psf_list: Optional[Sequence[PSFContainer]], prm: ParamMapping) -> None:
    """Replace ``$LIGAND`` constraints with explicit ligand atoms."""
    not_implemented = psf_list is None or len(psf_list) != 1
    if not not_implemented:
        psf: PSFContainer = psf_list[0]  # type: ignore[index]
        lig_id = psf.residue_id.iloc[-1]
        df = psf.atoms[psf.residue_id == lig_id]

        atom_types: Iterable[str] = df['atom type'].values
        atom_counter = Counter(atom_types)
        metadata = {'min': -np.inf, 'max': np.inf, 'count': 0, 'frozen': True,
                    'guess': False, 'unit': ''}
        for k in atom_counter:
            key = ('charge', 'charge', k)
            if key not in prm.param.index:
                value = df.loc[df['atom type'] == k, 'charge'].iloc[0]
                prm.add_param(key, value, **metadata)

    for lst in prm.constraints.values():
        if lst is None:
            continue
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
def update_count(
    param: ParamMapping, psf: Sequence[PSFContainer], mol: None | Sequence[MultiMolecule]
) -> None: ...
@overload
def update_count(param: ParamMapping, psf: None, mol: Sequence[MultiMolecule]) -> None: ...
def update_count(param, psf=None, mol=None):  # noqa: E302
    """Assign atom-counts to the passed :class:`ParamMapping`."""
    # Construct a generator
    if psf is not None:
        psf_iter = psf if len(psf) != 1 else repeat(psf[0], len(param.param.columns))
        count_iter: Iterator[Mapping[str, int]] = (pd.value_counts(p.atom_type) for p in psf_iter)
    elif mol is not None:
        mol_iter = mol if len(mol) != 1 else repeat(mol[0], len(param.param.columns))
        count_iter = ({k: len(v) for k, v in m.atoms.items()} for m in mol_iter)
    else:
        raise TypeError("'psf' and 'mol' cannot be both 'None'")

    index = param.metadata.index
    prm_count_list = [param.metadata[i, 'count'] for i in param.metadata.columns.levels[0]]
    at_sequence = [atoms.split() for *_, atoms in index]
    for count in count_iter:
        data = get_atom_count(at_sequence, count)
        series = pd.Series({k: v for k, v in zip(index, data) if v is not None}, name='unit')
        for prm_count in prm_count_list:
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
