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
    get_rdf_lowmem, get_rdf,
    get_adf,
    get_template, assert_error, get_example_xyz,
    update_charge, get_charge_constraints
)

from .io import (
    read_kf,
    read_multi_xyz,
    read_pdb,
    read_prm, write_prm, rename_atom_types,
    create_hdf5, create_xyz_hdf5, to_hdf5, from_hdf5
)

from .classes import (
    MultiMolecule,
    ARMC,
    Molecule,
    PSF
)

__version__ = __version__
__author__ = "Bas van Beek"
__email__ = 'b.f.van.beek@vu.nl'

__all__ = [
    'get_rdf_lowmem', 'get_rdf',
    'get_adf',
    'get_template', 'assert_error', 'get_example_xyz',

    'read_kf',
    'read_multi_xyz',
    'read_pdb',
    'read_prm', 'write_prm', 'rename_atom_types',
    'create_hdf5', 'create_xyz_hdf5', 'to_hdf5', 'from_hdf5',

    'MultiMolecule',
    'ARMC',
    'Molecule',
    'PSF'
]
