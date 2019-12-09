.. _Multi-XYZ reader:

Multi-XYZ reader
================

A reader of multi-xyz files has been implemented in the
:mod:`FOX.io.read_xyz` module. The .xyz fileformat is designed
for storing the atomic symbols and cartesian coordinates of one or more
molecules. The herein implemented :func:`FOX.io.read_xyz.read_multi_xyz`
function allows for the fast, and memory-effiecient, retrieval of the
various molecular geometries stored in an .xyz file.

An .xyz file, ``example_xyz_file``, can also be directly converted into
a :class:`.MultiMolecule` instance.

.. code:: python

    >>> from FOX import MultiMolecule, example_xyz

    >>> mol = MultiMolecule.from_xyz(example_xyz)

    >>> print(type(mol))
    <class 'FOX.classes.multi_mol.MultiMolecule'>


API
---

.. autofunction:: FOX.io.read_xyz.read_multi_xyz

.. automethod:: FOX.classes.multi_mol.MultiMolecule.from_xyz
    :noindex:

.. autodata:: FOX.example_xyz
