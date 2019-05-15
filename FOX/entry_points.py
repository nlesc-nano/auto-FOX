#!/usr/bin/env python
"""Entry points for Auto-FOX."""

import argparse
from typing import Optional
from os.path import isfile

from FOX import ARMC

__all__: list = []


def main_armc(args: Optional[list] = None) -> None:
    """Entrypoint for :meth:`FOX.classes.armc.ARMC.init_armc`."""
    parser = argparse.ArgumentParser(
         prog='FOX',
         usage='init_armc filename',
         description="Initalize the Auto-FOX Addaptive Rate Monte Carlo (ARMC) parameter optimizer.\
         See 'https://auto-fox.readthedocs.io/en/latest/4_monte_carlo.html' for a more detailed \
         description."
    )

    parser.add_argument(
        'filename', nargs='+', type=str, help='A .yaml file with ARMC settings'
    )

    filename = parser.parse_args(args).filename[0]
    if not isfile(filename):
        raise FileNotFoundError("[Errno 2] No such file: '{}'".format(filename))

    armc = ARMC.from_yaml(filename)
    armc.init_armc()
