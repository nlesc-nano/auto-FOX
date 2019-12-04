"""
Auto-FOX
========

A library for analyzing potential energy surfaces (PESs) and
using the resulting PES descriptors for constructing forcefield parameters.

Documentation
-------------
https://auto-fox.readthedocs.io/en/latest/

"""

from .__version__ import __version__

from .functions import (
    assert_error, get_example_xyz
)

from .io import (
    PSFContainer,
    PRMContainer,
    create_hdf5, create_xyz_hdf5, to_hdf5, from_hdf5
)

from .classes import (
    FrozenSettings,
    MultiMolecule,
    ARMC, run_armc,
)

from .ff import (
    get_bonded,
    get_non_bonded,
    get_intra_non_bonded,
    estimate_lj
)

__author__ = "Bas van Beek"
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    '__version__',

    'get_example_xyz', 'assert_error',

    'PSFContainer',
    'PRMContainer',
    'create_hdf5', 'create_xyz_hdf5', 'to_hdf5', 'from_hdf5',

    'FrozenSettings',
    'MultiMolecule',
    'ARMC', 'run_armc',

    'estimate_lj',
    'get_non_bonded',
    'get_intra_non_bonded',
    'get_bonded'
]
