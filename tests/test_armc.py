"""Tests for various ARMC jobs."""

from pathlib import Path
from typing import Tuple, Generator, Any, cast
from itertools import combinations_with_replacement

import numpy as np
import h5py
import yaml
from nanoutils import delete_finally
from assertionlib import assertion

from FOX.testing_utils import load_results
from FOX.armc import dict_to_armc, run_armc

PATH = Path('tests') / 'test_files'
ARMC_REF = PATH / 'ARMC_ref'
ARMCPT_REF = PATH / 'ARMCPT_ref'


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
        assertion.eq(f1.keys(), f2.keys())

        skip = {'param', 'aux_error_mod'}
        iterator = ((k, f1[k][:], f2[k][:]) for k in f2.keys() if k not in skip)
        for k, ar1, ar2 in iterator:
            np.testing.assert_allclose(ar1, ar2, err_msg=f'dataset {k!r}\n')


def swap_phi(*args: Any, n: int = 3, **kwargs: Any) -> Generator[Tuple[int, int], None, None]:
    while True:
        iterator = combinations_with_replacement(range(n), r=2)
        for i in iterator:
            yield cast(Tuple[int, int], i)


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
    with h5py.File(hdf5, 'r') as f1, h5py.File(hdf5_ref, 'r') as f2:
        assertion.eq(f1.keys(), f2.keys())

        skip = {'param', 'aux_error_mod'}
        iterator = ((k, f1[k][:], f2[k][:]) for k in f2.keys() if k not in skip)
        for k, ar1, ar2 in iterator:
            np.testing.assert_allclose(ar1, ar2, err_msg=f'dataset {k!r}\n')
