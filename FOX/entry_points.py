# #!/usr/bin/env python

""" Entry points for Auto-FOX. """

__all__ = []

import argparse
from os.path import isfile

from scm.plams import add_to_class
from scm.plams.interfaces.thirdparty.cp2k import Cp2kJob

from FOX import ARMC


def main_armc(args=None):
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
