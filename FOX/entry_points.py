#!/usr/bin/env python
"""Entry points for Auto-FOX.

Index
-----
.. currentmodule:: FOX.entry_points
.. autosummary::
    main_armc
    main_armc2yaml

API
---
.. autofunction:: main_armc
.. autofunction:: main_armc2yaml

"""

import os
import argparse
from os.path import isfile, join
from typing import Optional

import yaml
from nanoutils import UniqueLoader

from .armc import dict_to_armc, run_armc


def main_armc(args: Optional[list] = None) -> None:
    """Entrypoint for :func:`FOX.classes.armc.run_armc`."""
    parser = argparse.ArgumentParser(
        prog='FOX',
        usage='init_armc filename -r restart',
        description=("Initalize the Auto-FOX Addaptive Rate Monte Carlo (ARMC) "
                     "parameter optimizer."
                     "See 'https://auto-fox.readthedocs.io/en/latest/4_monte_carlo.html' for "
                     "a more detailed description.")
    )

    parser.add_argument(
        'filename', nargs=1, type=str, help='A .yaml file with ARMC settings.'
    )

    parser.add_argument(
        '-r', '--restart', nargs=1, type=bool, default=[False], required=False,
        help='Whether or not to continue a previous calculation.'
    )

    args_parsed = parser.parse_args(args)
    filename: str = args_parsed.filename[0]
    restart: bool = args_parsed.restart[0]

    with open(filename, 'r') as f:
        dct = yaml.load(f.read(), Loader=UniqueLoader)
    armc, kwargs = dict_to_armc(dct)
    run_armc(armc, restart=restart, **kwargs)


def main_armc2yaml(args: Optional[list] = None) -> None:
    """Entrypoint for :meth:`FOX.classes.armc.ARMC.to_yaml`."""
    parser = argparse.ArgumentParser(
        prog='FOX',
        usage='armc2yaml filename -o output',
        description="Convert an ARMC .yaml file into a pre-processed .yaml file"
    )

    parser.add_argument(
        'filename', nargs=1, type=str, help='A .yaml file with ARMC settings.'
    )

    parser.add_argument(
        '-o', '--output', nargs=1, type=str, metavar='output', required=False, default=[None],
        help=('Optional: the path+filename of the output .yaml file; '
              'will default to current working directory if not specified')
    )

    args_parsed = parser.parse_args(args)
    filename: str = args_parsed.filename[0]
    if not isfile(filename):
        raise FileNotFoundError(f"[Errno 2] No such file: {filename!r}")

    if args_parsed.output[0] is not None:
        output: str = args_parsed.output[0]
    else:  # Avoid duplicate names
        _output: str = join(os.getcwd(), 'armc.{:d}.yaml')
        i = 0
        while True:
            output = _output.format(i)
            if not isfile(output):
                break
            i += 1

    with open(filename, 'r') as f:
        dct_inp = yaml.safe_load(f.read())
    armc, kwargs = dict_to_armc(dct_inp)

    dct_out = armc.to_yaml_dict(
        path=kwargs['path'],
        folder=kwargs['folder'],
        logfile=kwargs['logfile'],
        psf=kwargs['psf'],
    )
    with open(output, 'w') as f:
        f.write(yaml.dump(dct_out))
