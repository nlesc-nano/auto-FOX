"""Tests for various ARMC jobs."""

import warnings
import pytest
import shutil
from os import PathLike
from pathlib import Path
from typing import Tuple, Generator, Any, cast, List, Optional, Union
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import h5py
import yaml
from nanoutils import delete_finally
from assertionlib import assertion

from FOX.testing_utils import load_results, compare_hdf5
from FOX.armc import dict_to_armc, run_armc
from FOX.armc.sanitization import _sort_atoms
from FOXdata import (
    ARMC_DIR as ARMC_REF,
    ARMCPT_DIR as ARMCPT_REF,
)

PATH = Path('tests') / 'test_files'

DF = pd.DataFrame({
    "param_type": ['charge', 'charge', 'charge', 'charge', 'epsilon', 'epsilon', 'epsilon', 'epsilon', 'epsilon', 'sigma', 'sigma', 'sigma', 'sigma', 'sigma'],  # noqa: E501
    "atoms": ['Cd', 'Se', 'O_1', 'C_1', 'Cd Cd', 'Se Se', 'Cd Se', 'Cd O_1', 'O_1 Se', 'Cd Cd', 'Se Se', 'Cd Se', 'Cd O_1', 'O_1 Se'],  # noqa: E501
})


def _prepare_restart(
    inp: Union[str, "PathLike[str]"],
    out: Union[str, "PathLike[str]"],
    super_iter: Optional[int] = None,
    sub_iter: Optional[int] = None
) -> None:
    shutil.copytree(inp, out)
    hdf5 = Path(out) / 'armc.hdf5'
    with h5py.File(hdf5, 'r+') as f:
        if super_iter is not None:
            f.attrs['super-iteration'] = super_iter
        if sub_iter is not None:
            f.attrs['sub-iteration'] = sub_iter


@delete_finally(PATH / '_ARMC')
@pytest.mark.parametrize('name', ['armc_ref.yaml'])
def test_armc_restart(name: str) -> None:
    """Test the restart functionality of :class:`ARMC` and :class:`ARMCPT`."""
    file = PATH / name
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    sub_iter = 5
    super_iter = 8
    i = 2 + (super_iter * 10 + sub_iter)

    armc, job_kwargs = dict_to_armc(dct)
    if name == 'armc_ref.yaml':
        REF = ARMC_REF
        armc.package_manager.hook = iter(load_results(ARMC_REF, n=1)[i:])
    else:
        iterator = _get_phi_iter()

        def swap_phi(*args: Any, **kwargs: Any) -> List[Tuple[int, int]]:
            return next(iterator)

        REF = ARMCPT_REF
        armc.swap_phi = swap_phi
        armc.package_manager.hook = iter(load_results(ARMCPT_REF, n=3)[i:])

    _prepare_restart(REF, PATH / job_kwargs['folder'], super_iter, sub_iter)
    hdf5_ref = REF / 'armc.hdf5'
    hdf5 = PATH / job_kwargs['folder'] / 'armc.hdf5'

    run_armc(armc, restart=True, **job_kwargs)
    assertion.assert_(next, armc.package_manager.hook, exception=StopIteration)
    with h5py.File(hdf5, 'r') as f1, h5py.File(hdf5_ref, 'r') as f2:
        compare_hdf5(f1, f2, skip={'/param', '/aux_error_mod'})
        assertion.shape_eq(f1['param'], f2['param'])
        assertion.shape_eq(f1['aux_error_mod'], f2['aux_error_mod'])
        assertion.eq(f1['param'].dtype, f2['param'].dtype)
        assertion.eq(f1['aux_error_mod'].dtype, f2['aux_error_mod'].dtype)


def test_armc_guess() -> None:
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    sigma = dct['param']['lennard_jones'][1]
    del sigma['Cd Cd']
    sigma['frozen'] = {'guess': 'uff'}

    armc, job_kwargs = dict_to_armc(dct)
    param = armc.param['param'].loc[('lennard_jones', 'sigma'), 0]

    # The expected `sigma` parameters
    ref = np.array([
        3.05043721,
        3.65491199,
        2.53727955,
        2.07044862,
        2.7831676,
        3.14175433,
        2.6749234,
        3.38764238,
        3.74622911,
    ])
    np.testing.assert_allclose(param.values, ref)


def test_allow_non_existent() -> None:
    """Test ``param.validation.allow_non_existent``."""
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Test `param.validation.allow_non_existent`
    dct['param']['charge']['Cl'] = -1
    assertion.assert_(dict_to_armc, dct, exception=RuntimeError)

    del dct['psf']
    del dct['param']['charge']['constraints'][0]
    assertion.assert_(dict_to_armc, dct, exception=RuntimeError)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dct['param']['validation']['allow_non_existent'] = True
        dct['param']['validation']['charge_tolerance'] = 'inf'
        assertion.assert_(dict_to_armc, dct)


def test_charge_tolerance() -> None:
    """Test ``param.validation.charge_tolerance``."""
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    dct = dct.copy()
    dct['param']['charge']['Cd'] = 2
    assertion.assert_(dict_to_armc, dct, exception=ValueError)

    dct['param']['validation']['charge_tolerance'] = 'inf'
    assertion.assert_(dict_to_armc, dct)


def _get_phi_iter(n: int = 3) -> Generator[List[Tuple[int, int]], None, None]:
    while True:
        iterator = combinations_with_replacement(range(n), r=2)
        for i in iterator:
            yield cast(List[Tuple[int, int]], [i])


@delete_finally(PATH / '_ARMC', PATH / '_ARMCPT')
@pytest.mark.parametrize('name', ['armc_ref.yaml', 'armcpt_ref.yaml'])
def test_armc(name: str) -> None:
    """Test :class:`ARMC` and :class:`ARMCPT`."""
    file = PATH / name
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    armc, job_kwargs = dict_to_armc(dct)
    if name == 'armc_ref.yaml':
        REF = ARMC_REF
        armc.package_manager.hook = iter(load_results(ARMC_REF, n=1))
    else:
        iterator = _get_phi_iter()

        def swap_phi(*args: Any, **kwargs: Any) -> List[Tuple[int, int]]:
            return next(iterator)

        REF = ARMCPT_REF
        armc.swap_phi = swap_phi
        armc.package_manager.hook = iter(load_results(ARMCPT_REF, n=3))

    hdf5_ref = REF / 'armc.hdf5'
    hdf5 = PATH / job_kwargs['folder'] / 'armc.hdf5'

    run_armc(armc, restart=False, **job_kwargs)
    assertion.assert_(next, armc.package_manager.hook, exception=StopIteration)
    with h5py.File(hdf5, 'r') as f1, h5py.File(hdf5_ref, 'r') as f2:
        compare_hdf5(f1, f2, skip={'/param', '/aux_error_mod'})
        assertion.shape_eq(f1['param'], f2['param'])
        assertion.shape_eq(f1['aux_error_mod'], f2['aux_error_mod'])
        assertion.eq(f1['param'].dtype, f2['param'].dtype)
        assertion.eq(f1['aux_error_mod'].dtype, f2['aux_error_mod'].dtype)


def test_param_sorting() -> None:
    """Tests for :func:`FOX.armc.sanitization._sort_atoms`"""
    df1 = DF.copy()
    df1.loc[6, "atoms"] = "Se Cd"
    df1.loc[13, "atoms"] = "Se O_1"
    _sort_atoms(df1)
    np.testing.assert_array_equal(df1, DF)

    df2 = DF.copy()
    df2.loc[4, "atoms"] = "Se Cd"
    df2.loc[5, "atoms"] = "Cd Se"
    df2.loc[6, "atoms"] = "Se Cd"
    assertion.assert_(_sort_atoms, df2, exception=KeyError)


@delete_finally(PATH / '_ARMC', PATH / '_ARMCPT')
@pytest.mark.parametrize('name', ['armc_ref.yaml', 'armcpt_ref.yaml'])
def test_yaml_armc(name: str) -> None:
    """Tests for :meth:`FOX.armc.ARMC.to_yaml_dict`."""
    file = PATH / name
    with open(file, 'r') as f:
        _dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    armc1, job_kwargs1 = dict_to_armc(_dct)
    dct1 = armc1.to_yaml_dict(
        path=job_kwargs1['path'],
        folder=job_kwargs1['folder'],
        logfile=job_kwargs1['logfile'],
        psf=job_kwargs1['psf'],
    )

    armc2, job_kwargs2 = dict_to_armc(dct1)
    dct2 = armc2.to_yaml_dict(
        path=job_kwargs2['path'],
        folder=job_kwargs2['folder'],
        logfile=job_kwargs2['logfile'],
        psf=job_kwargs2['psf'],
    )

    assertion.eq(armc1, armc2)
    assertion.eq(dct1, dct2)
