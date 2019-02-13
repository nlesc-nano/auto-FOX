
.. image:: https://travis-ci.org/BvB93/auto-FOX.svg?branch=master
   :target: https://travis-ci.org/BvB93/auto-FOX
.. image:: https://img.shields.io/badge/python-3.6-blue.svg
   :target: https://www.python.org

###########################################
Automated Forcefield Optimization Extension
###########################################

Auto-FOX tool for parameterizing forcefields by reproducing radial distribution functions.

Installation
============

Install Auto-FOX using pip:

- ``pip install git+https://github.com/BvB93/CAT@master#egg=CAT-0.1.0``


Currently implemented
=====================

This package is a work in progress; the following modules are currently implemented:

- A multi-xyz reader at FOX.functions.read_xyz_
- A radial distribution generator at FOX.functions.radial_distribution_

.. _FOX.functions.read_xyz: https://github.com/BvB93/auto-FOX/FOX/functions/read_xyz.py
.. _FOX.functions.radial_distribution: https://github.com/BvB93/auto-FOX/FOX/functions/radial_distribution.py
