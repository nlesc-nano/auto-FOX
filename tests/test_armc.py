"""Tests for various ARMC jobs."""

import warnings
from pathlib import Path
from typing import Tuple, Generator, Any, cast, Container, List
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import h5py
import yaml
from nanoutils import delete_finally
from assertionlib import assertion

from FOX.testing_utils import load_results
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


def compare_hdf5(f1: h5py.Group, f2: h5py.Group, skip: Container[str] = frozenset({})) -> None:
    """Check if the two passed hdf5 files are equivalent."""
    assertion.eq(f1.keys(), f2.keys())

    iterator1 = ((k, f1[k], f2[k]) for k in f2.keys() if k not in skip)
    for k1, dset1, dset2 in iterator1:
        if issubclass(dset1.dtype.type, np.inexact):
            np.testing.assert_allclose(dset1[:], dset2[:], err_msg=f'dataset {k1!r}\n')
        else:
            np.testing.assert_array_equal(dset1[:], dset2[:], err_msg=f'dataset {k1!r}\n')

        # Compare attributes
        assertion.eq(dset1.attrs.keys(), dset2.attrs.keys())
        iterator2 = ((k2, dset1.attrs[k2], dset1.attrs[k2]) for k2 in dset1.attrs.keys())
        for k2, attr1, attr2 in iterator2:
            err_msg = f'dataset {k1!r}; attribute {k2!r}'
            if issubclass(attr1.dtype.type, np.inexact):
                np.testing.assert_allclose(attr1, attr2, err_msg=err_msg)
            else:
                np.testing.assert_array_equal(attr1, attr2, err_msg=err_msg)


@delete_finally(PATH / '_ARMC')
def test_armc() -> None:
    """Test :class:`ARMC`."""
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    armc, job_kwargs = dict_to_armc(dct)
    armc.package_manager.hook = iter(load_results(ARMC_REF, n=1))

    run_armc(armc, restart=False, **job_kwargs)

    hdf5 = PATH / '_ARMC' / 'armc.hdf5'
    hdf5_ref = ARMC_REF / 'armc.hdf5'
    with h5py.File(hdf5, 'r') as f1, h5py.File(hdf5_ref, 'r') as f2:
        compare_hdf5(f1, f2, skip={'param', 'aux_error_mod'})
        assertion.shape_eq(f1['param'], f2['param'])
        assertion.shape_eq(f1['aux_error_mod'], f2['aux_error_mod'])
        assertion.eq(f1['param'].dtype, f2['param'].dtype)
        assertion.eq(f1['aux_error_mod'].dtype, f2['aux_error_mod'].dtype)


def _get_phi_iter(n: int = 3) -> Generator[List[Tuple[int, int]], None, None]:
    while True:
        iterator = combinations_with_replacement(range(n), r=2)
        for i in iterator:
            yield cast(List[Tuple[int, int]], [i])


ITERATOR = _get_phi_iter()


def swap_phi(*args: Any, **kwargs: Any) -> List[Tuple[int, int]]:
    return next(ITERATOR)


@delete_finally(PATH / '_ARMCPT')
def test_armcpt() -> None:
    """Test :class:`ARMC`."""
    file = PATH / 'armcpt_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    armc, job_kwargs = dict_to_armc(dct)
    armc.swap_phi = swap_phi
    armc.package_manager.hook = iter(load_results(ARMCPT_REF, n=3))

    run_armc(armc, restart=False, **job_kwargs)

    hdf5 = PATH / '_ARMCPT' / 'armc.hdf5'
    hdf5_ref = ARMCPT_REF / 'armc.hdf5'
    with h5py.File(hdf5, 'r') as f1, h5py.File(hdf5_ref, 'r+') as f2:
        compare_hdf5(f1, f2, skip={'param', 'aux_error_mod'})
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
