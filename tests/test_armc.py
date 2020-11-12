"""Tests for various ARMC jobs."""

from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import yaml
from nanoutils import delete_finally
from assertionlib import assertion

from FOX.testing_utils import load_results
from FOX.armc import dict_to_armc, run_armc
from FOX.armc.sanitization import _sort_atoms

PATH = Path('tests') / 'test_files'
REF_PATH = PATH / 'ARMC_ref'

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
        0.2471,
        3.14175433,
        2.6749234,
        3.38764238,
        0.3526,
        3.74622911,
    ])
    np.testing.assert_allclose(param.values, ref)


@delete_finally(PATH / '_ARMC')
def test_armc() -> None:
    """Test :class:`ARMC`."""
    file = PATH / 'armc_ref.yaml'
    with open(file, 'r') as f:
        dct = yaml.load(f.read(), Loader=yaml.FullLoader)

    armc, job_kwargs = dict_to_armc(dct)
    armc.package_manager.hook = iter(load_results(REF_PATH, n=1))

    run_armc(armc, restart=False, **job_kwargs)

    hdf5 = PATH / '_ARMC' / 'armc.hdf5'
    hdf5_ref = REF_PATH / 'armc.hdf5'
    with h5py.File(hdf5, 'r') as f1, h5py.File(hdf5_ref, 'r') as f2:
        assertion.eq(f1.keys(), f2.keys())

        skip = {'param', 'aux_error_mod'}
        iterator = ((k, f1[k][:], f2[k][:]) for k in f2.keys() if k not in skip)
        for k, ar1, ar2 in iterator:
            np.testing.assert_allclose(ar1, ar2, err_msg=f'dataset {k!r}\n')


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
