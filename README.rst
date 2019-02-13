
.. image:: https://travis-ci.org/BvB93/auto-FOX.svg?branch=master
   :target: https://travis-ci.org/BvB93/auto-FOX
.. image:: https://img.shields.io/badge/python-3-blue.svg
   :target: https://www.python.org

###########################################
Automated Forcefield Optimization Extension
###########################################

**Auto-FOX** is a tool for parameterizing forcefields by reproducing radial distribution functions.

Installation
============

Anaconda environments
---------------------

- While not a strictly required, it stronly recomended to use the virtual environments of `Anaconda <https://www.anaconda.com/>`_.


  - Available as either Miniconda_ or the complete `Anaconda <https://www.anaconda.com/distribution/#download-section>`_ package.


- Anaconda installation instructions are available for a wide range of OSs.


  - See the `Anaconda documentation <https://docs.anaconda.com/anaconda/install/>`_ for more details.   


- The anaconda environment can be created, enabled and disabled by, respectively, typing:

  - Create environment: ``conda create --name FOX python`` 

  - Enable environment: ``conda activate FOX`` 
  
  - Disable environment: ``conda deactivate``
    

Installing **Auto-FOX**
-----------------------

-  If using Conda, enable the environment: ``conda activate FOX`` 

-  Install **Auto-FOX** using pip: ``pip install git+https://github.com/BvB93/auto-FOX@master#egg=Auto-FOX-0.1.0``

-  An example input file is provided in the FOX.examples_ directory.


Currently implemented
=====================

This package is a work in progress; the following modules are currently implemented:

- A multi-xyz reader at FOX.functions.read_xyz_
- A radial distribution generator at FOX.functions.radial_distribution_

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _FOX.examples: https://github.com/BvB93/auto-FOX/blob/master/FOX/examples/input.py
.. _FOX.functions.read_xyz: https://github.com/BvB93/auto-FOX/tree/master/FOX/functions/read_xyz.py
.. _FOX.functions.radial_distribution: https://github.com/BvB93/auto-FOX/tree/master/FOX/functions/radial_distribution.py
