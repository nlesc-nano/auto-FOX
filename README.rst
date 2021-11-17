
.. image:: https://github.com/nlesc-nano/auto-FOX/workflows/Tests/badge.svg
    :target: https://github.com/nlesc-nano/auto-FOX/actions?query=workflow%3ATests+branch%3Amaster
.. image:: https://readthedocs.org/projects/auto-fox/badge/?version=latest
    :target: https://auto-fox.readthedocs.io/en/latest/
.. image:: https://codecov.io/gh/nlesc-nano/auto-FOX/branch/master/graph/badge.svg?token=7IgHsRDVdo
    :target: https://codecov.io/gh/nlesc-nano/auto-FOX
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3988142.svg
    :target: https://doi.org/10.5281/zenodo.3988142

|

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: https://docs.python.org/3.7/
.. image:: https://img.shields.io/badge/python-3.8-blue.svg
    :target: https://docs.python.org/3.8/
.. image:: https://img.shields.io/badge/python-3.9-blue.svg
    :target: https://docs.python.org/3.9/

##################################################
Automated Forcefield Optimization Extension 0.10.0
##################################################

**Auto-FOX** is a library for analyzing potential energy surfaces (PESs) and
using the resulting PES descriptors for constructing forcefield parameters.
Further details are provided in the documentation_.


Currently implemented
=====================

This package is a work in progress; the following
functionalities are currently implemented:

- The MultiMolecule class, a class designed for handling and processing
  potential energy surfaces. (1_)
- A multi-XYZ reader. (2_)
- A radial and angular distribution generator (RDF & ADF). (3_)
- A root mean squared displacement generator (RMSD). (4_)
- A root mean squared fluctuation generator (RMSF). (5_)
- Tools for describing shell structures in, *e.g.*,
  nanocrystals or dissolved solutes. (6_)
- A Monte Carlo forcefield parameter optimizer. (7_)

Using **Auto-FOX**
==================

- An input file with some basic examples is provided in
  the FOX.examples_ directory.

- An example MD trajectory of a CdSe quantum dot is included
  in the FOX.data_ directory.

  - The absolute path + filename of aforementioned trajectory
    can be retrieved as following:


.. code:: python

    >>> from FOX import example_xyz

- Further examples and more detailed descriptions are
  available in the documentation_.


Installation
============

Anaconda environments
---------------------

- While not a strictly required, it stronly recomended to use the
  virtual environments of Anaconda.

  - Available as either Miniconda_ or the complete Anaconda_ package.

- Anaconda comes with a built-in installer; more detailed installation
  instructions are available for a wide range of OSs.

  - See the `Anaconda documentation <https://docs.anaconda.com/anaconda/install/>`_.

- Anaconda environments can be created, enabled and disabled by,
  respectively, typing:

  - Create environment: ``conda create -n FOX -c conda-forge python pip``

  - Enable environment: ``conda activate FOX``

  - Disable environment: ``conda deactivate``

Installing **Auto-FOX**
-----------------------

- If using Conda, enable the environment: ``conda activate FOX``

- Install **Auto-FOX** with PyPi: ``pip install auto-FOX --upgrade``

- Congratulations, **Auto-FOX** is now installed and ready for use!

Optional dependencies
---------------------

- The plotting of data produced by **Auto-FOX** requires Matplotlib_.
  Matplotlib is distributed by both PyPi and Anaconda:

  - Anaconda:   ``conda install --name FOX -y -c conda-forge matplotlib``

  - PyPi:       ``pip install matplotlib``

- Construction of the angular distribution function in parallel requires DASK_.

  - Anaconda:   ``conda install -name FOX -y -c conda-forge dask``

- RDKit is required for a number of .psf-related recipes.

  - Anaconda:   ``conda install -name FOX -y -c conda-forge rdkit``


.. _1: https://auto-fox.readthedocs.io/en/latest/3_multimolecule.html
.. _2: https://auto-fox.readthedocs.io/en/latest/5_xyz_reader.html
.. _3: https://auto-fox.readthedocs.io/en/latest/1_rdf.html
.. _4: https://auto-fox.readthedocs.io/en/latest/2_rmsd.html#root-mean-squared-displacement
.. _5: https://auto-fox.readthedocs.io/en/latest/2_rmsd.html#root-mean-squared-fluctuation
.. _6: https://auto-fox.readthedocs.io/en/latest/2_rmsd.html#discerning-shell-structures
.. _7: https://auto-fox.readthedocs.io/en/latest/4_monte_carlo.html
.. _8: https://www.youtube.com/watch?v=hFDcoX7s6rE
.. _documentation: https://auto-fox.readthedocs.io/en/latest/
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Anaconda: https://www.anaconda.com/distribution/#download-section
.. _Matplotlib: https://matplotlib.org/
.. _FOX.data: https://github.com/nlesc-nano/auto-FOX/blob/master/FOX/data
.. _FOX.examples: https://github.com/nlesc-nano/auto-FOX/blob/master/FOX/examples/input.py
.. _DASK: https://dask.org/
.. _RDKit: https://www.rdkit.org/
