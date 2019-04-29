#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit Auto-FOX/__version__.py
version = {}
with open(os.path.join(here, 'FOX', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='Auto-FOX',
    version=version['__version__'],
    description='A tool for parameterizing forcefields by reproducing radial distribution functions.',
    long_description=readme + '\n\n',
    author='Bas van Beek',
    author_email='b.f.van.beek@vu.nl',
    url='https://github.com/nlesc-nano/Auto-FOX',
    packages=[
        'FOX',
        'FOX.data',
        'FOX.examples',
        'FOX.functions',
        'FOX.classes'
    ],
    package_dir={'FOX': 'FOX'},
    package_data={'FOX': [
        'data/*.xyz',
        'data/*.yaml',
        'data/*.str'
    ]},
    include_package_data=True,
    license="GNU General Public License v3 or later",
    zip_safe=False,
    keywords=[
        'quantum-mechanics',
        'molecular-mechanics',
        'science',
        'chemistry',
        'python-3',
        'python-3.7',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry'
        'License :: OSI Approved :: GNU Lesser General Public License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    test_suite='tests',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'plams@git+https://github.com/SCM-NV/PLAMS@master'
    ],
    setup_requires=[
        'pytest-runner',
        'sphinx',
        'sphinx_rtd_theme',
        'recommonmark'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'pycodestyle',
    ],
    extras_require={
        'MC': ['pyyaml', 'schema'],
        'doc': ['sphinx', 'sphinx_rtd_theme', 'nbsphinx', 'matplotlib'],
        'test': ['pytest', 'pytest-cov', 'pytest-mock', 'nbsphinx',
                 'pycodestyle', 'pyyaml', 'matplotlib'],
    }
)
