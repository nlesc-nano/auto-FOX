
.. image:: https://travis-ci.org/BvB93/auto-FOX.svg?branch=master
   :target: https://travis-ci.org/BvB93/auto-FOX
.. image:: https://readthedocs.org/projects/auto-fox/badge/?version=latest
   :target: https://auto-fox.readthedocs.io/en/latest
.. image:: https://img.shields.io/badge/python-3-blue.svg
   :target: https://www.python.org

###########################################
Automated Forcefield Optimization Extension
###########################################

**Auto-FOX** is a tool for parameterizing forcefields by reproducing radial distribution functions.
Further details are provided in the documentation_.

Installation
============

Anaconda environments
---------------------

- While not a strictly required, it stronly recomended to use the virtual environments of `Anaconda <https://www.anaconda.com/>`_.


  - Available as either Miniconda_ or the complete `Anaconda <https://www.anaconda.com/distribution/#download-section>`_ package.


- Anaconda comes with a built-in installer; more installation instructions are available for a wide range of OSs.


  - See the `Anaconda documentation <https://docs.anaconda.com/anaconda/install/>`_ for more details.


- The anaconda environment can be created, enabled and disabled by, respectively, typing:

  - Create environment: ``conda create --name FOX python``

  - Enable environment: ``conda activate FOX``

  - Disable environment: ``conda deactivate``


Installing **Auto-FOX**
-----------------------

-  If using Conda, enable the environment: ``conda activate FOX``

-  Install **Auto-FOX** using pip: ``pip install git+https://github.com/BvB93/auto-FOX@master#egg=Auto-FOX-0.1.0``

-  Optional: For plotting graphs produced by **Auto-FOX**, install Matplotlib_ using either conda or pip: 

   -  ``conda install --name FOX -y -c conda-forge matplotlib`` or ``pip install matplotlib``

-  Congratulations, **Auto-FOX** is now installed and ready for use!

Using **Auto-FOX**
------------------

-  An input file with some basic examples is provided in the FOX.examples_ directory.

-  Further examples and more detailed descriptions are available in the documentation_.


Currently implemented
=====================

This package is a work in progress; the following functionalities are currently implemented:

- The MultiMolecule class, a class designed for handling and processing large amounts of moleculair conformations and/or configurations (*e.g.* MD trajectories). (`doc <https://auto-fox.readthedocs.io/en/latest/MultiMolecule.html>`_)
- A multi-XYZ reader. (`doc <https://auto-fox.readthedocs.io/en/latest/xyz_reader.html>`_)
- A radial distribution generator (RDF). (`doc <https://auto-fox.readthedocs.io/en/latest/RDF.html>`_)
- A root mean squared displacement generator (RMSD). (`doc <https://auto-fox.readthedocs.io/en/latest/RMSD.html#root-mean-squared-displacement>`_)
- A root mean squared fluctuation generator (RMSF). (`doc <https://auto-fox.readthedocs.io/en/latest/RMSD.html#root-mean-squared-fluctuation>`_)


.. _documentation: https://auto-fox.readthedocs.io/en/latest/
.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _Matplotlib: https://matplotlib.org/
.. _FOX.examples: https://github.com/BvB93/auto-FOX/blob/master/FOX/examples/input.py
