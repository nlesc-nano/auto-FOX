
.. image:: https://travis-ci.org/BvB93/auto-FOX.svg?branch=master
   :target: https://travis-ci.org/BvB93/auto-FOX
.. image:: https://readthedocs.org/projects/auto-fox/badge/?version=latest
   :target: https://auto-fox.readthedocs.io/en/latest
.. image:: https://img.shields.io/badge/python-3-blue.svg
   :target: https://www.python.org

#################################################
Automated Forcefield Optimization Extension 0.3.0
#################################################

**Auto-FOX** is a tool for parameterizing forcefields by reproducing
radial distribution functions.
Further details are provided in the documentation_.

Currently implemented
=====================

This package is a work in progress; the following
functionalities are currently implemented:

- The MultiMolecule class, a class designed for handling and processing
  large numbers of moleculair conformations and/or configurations
  (*e.g.* MD trajectories). (1_)
- A multi-XYZ reader. (2_)
- A radial and angular distribution generator (RDF & ADF). (3_)
- A root mean squared displacement generator (RMSD). (4_)
- A root mean squared fluctuation generator (RMSF). (5_)

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

  - Create environment: ``conda create --name FOX python``

  - Enable environment: ``conda activate FOX``

  - Disable environment: ``conda deactivate``


Installing **Auto-FOX**
-----------------------

- If using Conda, enable the environment: ``conda activate FOX``

- Install **Auto-FOX** using pip: ``pip install git+https://github.com/nlesc-nano/auto-FOX@master --upgrade``

- Congratulations, **Auto-FOX** is now installed and ready for use!

Optional dependencies
---------------------

-  For the plotting of data produced by **Auto-FOX** install Matplotlib_.
   Matplotlib can be installed with either conda or pip:

   -  Anaconda:   ``conda install --name FOX -y -c conda-forge matplotlib``

   -  PIP:        ``pip install matplotlib``

Using **Auto-FOX**
==================

- An input file with some basic examples is provided in
  the FOX.examples_ directory.

- An example MD trajectory of a CdSe quantum dot is included
  in the FOX.data_ directory.

   - The absolute path + filename of aforementioned trajectory
     can be retrieved as following:

::

         from FOX.examples.example_xyz import get_example_xyz
         example_xyz_filename = get_example_xyz()


- Further examples and more detailed descriptions are
  available in the documentation_.


.. _1: https://auto-fox.readthedocs.io/en/latest/3_multimolecule.html
.. _2: https://auto-fox.readthedocs.io/en/latest/5_xyz_reader.html
.. _3: https://auto-fox.readthedocs.io/en/latest/1_rdf.html
.. _4: https://auto-fox.readthedocs.io/en/latest/2_rmsd.html#root-mean-squared-displacement
.. _5: https://auto-fox.readthedocs.io/en/latest/2_rmsd.html#root-mean-squared-fluctuation
.. _documentation: https://auto-fox.readthedocs.io/en/latest/
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Anaconda: https://www.anaconda.com/distribution/#download-section
.. _Matplotlib: https://matplotlib.org/
.. _FOX.data: https://github.com/nlesc-nano/auto-FOX/blob/master/FOX/data
.. _FOX.examples: https://github.com/nlesc-nano/auto-FOX/blob/master/FOX/examples/input.py
