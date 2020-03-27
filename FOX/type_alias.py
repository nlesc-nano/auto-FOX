"""
FOX.type_alias
==============

A module with type-aliases for objects.

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
    'WorkflowManagerABC': 'FOX.armc.workflow_manager.WorkflowManagerABC',
    'WorkflowManager': 'FOX.armc.workflow_manager.WorkflowManager',
    'PhiUpdaterABC': 'FOX.armc.phi_updater.PhiUpdaterABC',
    'PhiUpdater': 'FOX.armc.phi_updater.PhiUpdater',
    'ParamMappingABC': 'FOX.armc.param_mapping.ParamMappingABC',
    'ParamMapping': 'FOX.armc.param_mapping.ParamMapping',

    'KFReader': 'scm.plams.tools.kftools.KFReader',
    'Figre': 'matplotlib.figure.Figure',
    'File': 'h5py._hl.files.File',
    'NDFrame': 'pandas.core.generic.NDFrame'
})

globals().update(TYPE_ALIAS)

__all__ = list(TYPE_ALIAS.keys())

__doc__ = __doc__.format(
    autosummary='\n'.join(f'    {i}' for i in __all__),
    autodata='\n'.join(f'.. autodata:: {i}' for i in __all__)
)
