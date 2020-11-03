"""A module with type-aliases for objects.

Index
-----
.. currentmodule:: FOX.type_alias
.. autosummary::
{autosummary}

API
---
{autodata}

"""

from types import MappingProxyType
from typing import Mapping

TYPE_ALIAS: Mapping[str, str] = MappingProxyType({
    'MultiMolecule': 'FOX.classes.multi_mol.MultiMolecule',
    'FrozenSettings': 'FOX.classes.frozen_settings.FrozenSettings',
    'PRMContainer': 'FOX.io.read_prm.PRMContainer',
    'PSFContainer': 'FOX.io.read_psf.PSFContainer',
    'MonteCarloABC': 'FOX.armc.monte_carlo.MonteCarloABC',
    'ARMC': 'FOX.armc.armc.ARMC',
    'PackageManagerABC': 'FOX.armc.package_manager.PackageManagerABC',
    'PackageManager': 'FOX.armc.package_manager.PackageManager',
    'PhiUpdaterABC': 'FOX.armc.phi_updater.PhiUpdaterABC',
    'PhiUpdater': 'FOX.armc.phi_updater.PhiUpdater',
    'ParamMappingABC': 'FOX.armc.param_mapping.ParamMappingABC',
    'ParamMapping': 'FOX.armc.param_mapping.ParamMapping',

    'PathLike': 'os.PathLike',

    'KFReader': 'scm.plams.tools.kftools.KFReader',
    'Job': 'scm.plams.core.basejob.Job',
    'SingleJob': 'scm.plams.core.basejob.SingleJob',
    'Settings': 'scm.plams.core.settings.Settings',
    'Molecule': 'scm.plams.mol.molecule.Molecule',
    'Results': 'scm.plams.core.results.Results',

    'Figure': 'matplotlib.figure.Figure',

    'File': 'h5py._hl.files.File',

    'NDFrame': 'pandas.core.generic.NDFrame',

    'Result': 'qmflows.packages.packages.Result',
    'Package': 'qmflows.packages.packages.Package',

    'Registry': 'noodles.serial.registry.Registry',
    'PromisedObject': 'noodles.interface.decorator.PromisedObject'
})

globals().update(TYPE_ALIAS)

__all__ = list(TYPE_ALIAS.keys())

__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autodata='\n'.join(f'.. autodata:: {i}' for i in __all__)
)
