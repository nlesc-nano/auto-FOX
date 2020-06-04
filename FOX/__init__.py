"""**Auto-FOX** is a library for analyzing potential energy surfaces (PESs) and using the resulting PES descriptors for constructing forcefield parameters.

Documentation
-------------
https://auto-fox.readthedocs.io/en/latest/

"""  # noqa: E501

from os.path import join
from nanoutils import VersionInfo

from scm.plams import Settings as _Settings

from .__version__ import __version__

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

__author__ = "Bas van Beek"
__email__ = 'b.f.van.beek@vu.nl'

if hasattr(_Settings, 'suppress_missing'):
    _Settings.supress_missing = _Settings.suppress_missing

#: The path+filename of the example multi-xyz file.
example_xyz: str = join(__path__[0], 'data', 'Cd68Se55_26COO_MD_trajec.xyz')
del join

version_info = VersionInfo.from_str(__version__)
del VersionInfo

__all__ = [
    'example_xyz',

    'PSFContainer',
    'PRMContainer',
    'create_hdf5', 'create_xyz_hdf5', 'to_hdf5', 'from_hdf5',

    'MultiMolecule',

    'estimate_lj', 'get_free_energy',
    'get_non_bonded',
    'get_intra_non_bonded',
    'get_bonded',

    'recipes'
]
