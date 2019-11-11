#!/usr/bin/env python
"""
FOX.entry_points
================

Entry points for Auto-FOX.

Index
-----
.. currentmodule:: FOX.entry_points
.. autosummary::
    main_armc
    main_plot_pes
    main_plot_param
    main_plot_dset
    main_dset_to_csv

API
---
.. autofunction:: FOX.entry_points.main_armc
.. autofunction:: FOX.entry_points.main_plot_pes
.. autofunction:: FOX.entry_points.main_plot_param
.. autofunction:: FOX.entry_points.main_plot_dset
.. autofunction:: FOX.entry_points.main_dset_to_csv

"""

import argparse
from os.path import isfile
from typing import Optional

from .classes.armc import ARMC
from .armc_functions.csv import dset_to_csv
from .armc_functions.plotting import plot_pes_descriptors, plot_param, plot_dset

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

try:
    import h5py
    H5PY_ERROR = ''
except ImportError:
    H5PY_ERROR = ("Use of the FOX.{} function requires the 'h5py' package."
                  "\n'h5py' can be installed via anaconda with the following command:"
                  "\n\tconda install --name FOX -y -c conda-forge h5py")

__all__: list = []


def main_armc(args: Optional[list] = None) -> None:
    """Entrypoint for :meth:`FOX.classes.armc.ARMC.init_armc`."""
    parser = argparse.ArgumentParser(
         prog='FOX',
         usage='init_armc filename',
         description=("Initalize the Auto-FOX Addaptive Rate Monte Carlo (ARMC) "
                      "parameter optimizer."
                      "See 'https://auto-fox.readthedocs.io/en/latest/4_monte_carlo.html' for "
                      "a more detailed description.")
    )

    parser.add_argument(
        'filename', nargs=1, type=str, help='A .yaml file with ARMC settings.'
    )

    filename = parser.parse_args(args).filename[0]
    if not isfile(filename):
        raise FileNotFoundError("[Errno 2] No such file: '{}'".format(filename))

    armc = ARMC.from_yaml(filename)
    armc.init_armc()


def main_plot_pes(args: Optional[list] = None) -> None:
    """Entrypoint for :func:`FOX.armc_functions.plotting.plot_pes_descriptors`."""
    parser = argparse.ArgumentParser(
         prog='FOX',
         usage='plot_pes input -o output -i iteration -dset dset1 dset2 ...',
         description='Create side by side plots of MM PES descriptors and QM PES descriptors'
    )

    parser.add_argument(
        'input', nargs=1, type=str, metavar='input',
        help='Rquired: The path+name of the ARMC .hdf5 file.'
    )

    parser.add_argument(
        '-o', '--output', nargs=1, type=str, metavar='output', required=False, default=[None],
        help=('Optional: The path+name of the to-be created .png file. '
              'Set to the PES descriptor name (appended with ".png") by default.')
    )

    parser.add_argument(
        '-i', '--iteration', nargs=1, type=int, default=[-1], required=False, metavar='iteration',
        help=('Optional: The ARMC iteration containing the PES descriptor of interest. '
              'Set to the last iteration by default.')
    )

    parser.add_argument(
        '-dset', '--datasets', nargs='+', type=str, metavar='datasets', required=True,
        dest='datasets',
        help=('Required: One or more hdf5 dataset names. '
              'The provided dataset(s) should containing PES descriptors.')
    )

    # Unpack arguments
    args_parsed = parser.parse_args(args)
    input_ = args_parsed.input[0]
    output = args_parsed.output[0]
    iteration = args_parsed.iteration[0]
    datasets = args_parsed.datasets
    if not datasets:
        raise ValueError('The "--datasets" argument expects one or more PES descriptor names')

    with h5py.File(input_, 'r', libver='latest') as f:
        datasets_ = []
        for key in datasets:
            if key in f.keys():
                datasets_.append(key)
            else:
                i = 0
                while i >= 0:
                    try:
                        assert f'{key}.{i}' in f.keys()
                        datasets_.append(f'{key}.{i}')
                        i += 1
                    except AssertionError:
                        i = -1

    if output is None:
        for dset in datasets_:
            fig = plot_pes_descriptors(input_, dset, dset + '.png', iteration=iteration)
            plt.show(block=True)
    else:
        for i, dset in enumerate(datasets_):
            fig = plot_pes_descriptors(input_, dset, str(i) + '_' + output, iteration=iteration)
            plt.show(block=True)


def main_plot_param(args: Optional[list] = None) -> None:
    """Entrypoint for :func:`FOX.armc_functions.plotting.plot_param`."""
    parser = argparse.ArgumentParser(
         prog='FOX',
         usage='plot_pes input -o output',
         description='Create side by side plots of MM PES descriptors and QM PES descriptors'
    )

    parser.add_argument(
        'input', nargs=1, type=str, metavar='input',
        help='Rquired: The path+name of the ARMC .hdf5 file.'
    )

    parser.add_argument(
        '-o', '--output', nargs=1, type=str, metavar='output', required=False, default=[None],
        help=('Optional: The path+name of the to-be created .png file. '
              'Set to "param.png" by default.')
    )

    # Unpack arguments
    args_parsed = parser.parse_args(args)
    input_ = args_parsed.input[0]
    output = args_parsed.output[0]

    if output is None:
        fig = plot_param(input_, 'param.png')
    else:
        fig = plot_param(input_, output)
    plt.show(block=True)


def main_plot_dset(args: Optional[list] = None) -> None:
    """Entrypoint for :func:`FOX.armc_functions.plotting.plot_dset`."""
    parser = argparse.ArgumentParser(
         prog='FOX',
         usage='plot_pes input -o output -dset dset1 dset2 ...',
         description=('Plot one or more arbitrary datasets. '
                      'See `plot_param` and `plot_pes` for commands specialized in '
                      'plotting parameters and PES descriptors, respectively.')
    )

    parser.add_argument(
        'input', nargs=1, type=str, metavar='input',
        help='Rquired: The path+name of the ARMC .hdf5 file.'
    )

    parser.add_argument(
        '-o', '--output', nargs=1, type=str, metavar='output', required=False, default=[None],
        help=('Optional: The path+name of the to-be created .png file. '
              'Set to the dataset name (appended with ".png") by default.')
    )

    parser.add_argument(
        '-dset', '--datasets', nargs='+', type=str, metavar='datasets', required=True,
        dest='datasets', help=('Required: One or more hdf5 dataset names.')
    )

    # Unpack arguments
    args_parsed = parser.parse_args(args)
    input_ = args_parsed.input[0]
    output = args_parsed.output[0]
    datasets = args_parsed.datasets
    if not datasets:
        raise ValueError('The "--datasets" argument expects one or more dataset names')

    if output is None:
        fig = plot_dset(input_, datasets, 'datasets.png')
    else:
        fig = plot_dset(input_, datasets, output)
    plt.show(block=True)


def main_dset_to_csv(args: Optional[list] = None) -> None:
    """Entrypoint for :func:`FOX.armc_functions.csv.dset_to_csv`."""
    parser = argparse.ArgumentParser(
         prog='FOX',
         usage='plot_pes input -o output -dset dset1 dset2 ...',
         description=('Plot one or more arbitrary datasets. '
                      'See `plot_param` and `plot_pes` for commands specialized in '
                      'plotting parameters and PES descriptors, respectively.')
    )

    parser.add_argument(
        'input', nargs=1, type=str, metavar='input',
        help='Rquired: The path+name of the ARMC .hdf5 file.'
    )

    parser.add_argument(
        '-o', '--output', nargs=1, type=str, metavar='output', required=False, default=[None],
        help=('Optional: The path+name of the to-be created .png file. '
              'Set to the dataset name (appended with ".csv") by default.')
    )

    parser.add_argument(
        '-i', '--iteration', nargs=1, type=int, default=[-1], required=False, metavar='iteration',
        help=('Optional: The ARMC iteration containing the PES descriptor of interest. '
              'Set to the last iteration by default. '
              'Only relevant for datasets with more than two dimensions; '
              'will be ignored otherwise.')
    )

    parser.add_argument(
        '-dset', '--datasets', nargs='+', type=str, metavar='datasets', required=True,
        dest='datasets', help=('Required: One or more hdf5 dataset names.')
    )

    # Unpack arguments
    args_parsed = parser.parse_args(args)
    input_ = args_parsed.input[0]
    output = args_parsed.output[0]
    iteration = args_parsed.iteration[0]
    datasets = args_parsed.datasets
    if not datasets:
        raise ValueError('The "--datasets" argument expects one or more dataset names')

    if output is None:
        for dset in datasets:
            dset_to_csv(input_, dset, dset + '.csv', iteration=iteration)
    else:
        for i, dset in enumerate(datasets):
            dset_to_csv(input_, dset, str(i) + '_' + output, iteration=iteration)
