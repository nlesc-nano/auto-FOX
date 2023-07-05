"""**Auto-FOX** is a library for analyzing potential energy surfaces (PESs) and using the resulting PES descriptors for constructing forcefield parameters.

Documentation
-------------
https://auto-fox.readthedocs.io/en/latest/

"""  # noqa: E501

import warnings
from os.path import join, dirname

import pandas as pd

from ._version import __version__, __version_tuple__

from .io import (
    PSFContainer,
    PRMContainer,
    RTFContainer,
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

from . import recipes, properties

#: The path+filename of the example multi-xyz file.
example_xyz: str = join(dirname(__file__), 'data', 'Cd68Se55_26COO_MD_trajec.xyz')
del join, dirname

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
del pd, warnings

__all__ = [
    '__version__', '__version_tuple__',
    'example_xyz',

    'PSFContainer',
    'PRMContainer',
    'RTFContainer',
    'create_hdf5', 'create_xyz_hdf5', 'to_hdf5', 'from_hdf5',

    'MultiMolecule',

    'estimate_lj', 'get_free_energy',
    'get_non_bonded',
    'get_intra_non_bonded',
    'get_bonded',

    'recipes', 'properties',
]
