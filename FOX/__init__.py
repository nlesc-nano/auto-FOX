"""
Auto-FOX
========

A library for analyzing potential energy surfaces (PESs) and
using the resulting PES descriptors for constructing forcefield parameters.

Documentation
-------------
https://auto-fox.readthedocs.io/en/latest/

"""

from os.path import join

from .__version__ import __version__

from .functions import (
    assert_error, get_example_xyz, group_by_values
)

from .io import (
    PSFContainer,
    PRMContainer,
    create_hdf5, create_xyz_hdf5, to_hdf5, from_hdf5
)

from .classes import (
    MultiMolecule
)

from .ff import (
    get_bonded,
    get_non_bonded,
    get_intra_non_bonded,
    estimate_lj, get_free_energy
)

from .armc import (
    ARMC, run_armc
)

__author__ = "Bas van Beek"
__email__ = 'b.f.van.beek@vu.nl'
__version__ = __version__

#: The path+filename of the example multi-xyz file.
example_xyz: str = join(__path__[0], 'data', 'Cd68Se55_26COO_MD_trajec.xyz')
del join

__all__ = [
    'get_example_xyz', 'example_xyz', 'assert_error', 'group_by_values',

    'PSFContainer',
    'PRMContainer',
    'create_hdf5', 'create_xyz_hdf5', 'to_hdf5', 'from_hdf5',

    'MultiMolecule',
    'ARMC', 'run_armc',

    'estimate_lj', 'get_free_energy',
    'get_non_bonded',
    'get_intra_non_bonded',
    'get_bonded',

    'recipes'
]
