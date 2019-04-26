.. _Multi-XYZ reader:

Multi-XYZ reader
================

A reader of multi-xyz files has been implemented in the
:mod:`FOX.functions.read_xyz` module. The .xyz fileformat is designed
for storing the atomic symbols and cartesian coordinates of one or more
molecules. The herein implemented
:func:`FOX.functions.read_xyz.read_multi_xyz`
function allows for the fast, and memory-effiecient, retrieval of the
various molecular geometries stored in an .xyz file.

An .xyz file, ``example_xyz_file``, can also be directly converted into
a :class:`FOX.MultiMolecule` object.

.. code:: python

    >>> from FOX import (MultiMolecule, get_example_xyz)

    >>> example_xyz_file = get_example_xyz()
    >>> mol = MultiMolecule.from_xyz(example_xyz_file)
    >>> type(mol)
    FOX.classes.multi_mol.MultiMolecule


API
---

.. autofunction:: FOX.functions.read_xyz.read_multi_xyz
