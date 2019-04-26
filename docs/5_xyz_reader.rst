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

An .xyz file, ``xyz_file``, can also be directly converted into
a *MultiMolecule* object: ``rdf = MultiMolecule(input=xyz)``.

API
---

.. autofunction:: FOX.functions.read_xyz.read_multi_xyz
