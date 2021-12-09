#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit Auto-FOX/__version__.py
version = {}
with open(os.path.join(here, 'FOX', '__version__.py'), encoding='utf-8') as f:
    exec(f.read(), version)

with open('README.rst', encoding='utf-8') as readme_file:
    readme = readme_file.read()

docs_require = [
    'sphinx>=2.4',
    'sphinx_rtd_theme>=0.3.0',
    'matplotlib>=3.0',
]

tests_require_no_optional = [
    'pytest>=5.4.0',
    'pytest-cov',
]
tests_require = [
    'ase>=3.21.1',
    'nlesc-CAT>=0.10.0',
]
tests_require += tests_require_no_optional
tests_require += docs_require

setup(
    name='Auto-FOX',
    version=version['__version__'],
    description=('A library for analyzing potential energy surfaces (PESs) and using the resulting'
                 ' PES descriptors for constructing forcefield parameters.'),
    long_description=readme + '\n\n',
    author='Bas van Beek',
    author_email='b.f.van.beek@vu.nl',
    url='https://github.com/nlesc-nano/Auto-FOX',
    packages=[
        'FOX',
        'FOX.data',
        'FOX.examples',
        'FOX.functions',
        'FOX.armc',
        'FOX.classes',
        'FOX.io',
        'FOX.ff',
        'FOX.recipes',
        'FOX.properties',
    ],
    package_dir={'FOX': 'FOX'},
    package_data={'FOX': [
        'data/*.xyz',
        'data/*.yaml',
        'data/*.csv',
        'properties/*.pyi',
        'recipes/*.pyi',
        'py.typed',
        '*.pyi',
    ]},
    include_package_data=True,
    entry_points={'console_scripts': [
        'init_armc=FOX.entry_points:main_armc',
        'armc2yaml=FOX.entry_points:main_armc2yaml',
    ]},
    license="GNU General Public License v3 or later",
    zip_safe=False,
    keywords=[
        'quantum-mechanics',
        'molecular-mechanics',
        'science',
        'chemistry',
        'forcefield-parameterization',
        'forcefield',
        'python-3',
        'python-3.7',
        'python-3.8',
        'python-3.9',
        'python-3.10',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries ',
        'Typing :: Typed',
    ],
    python_requires='>=3.7',
    install_requires=[
        'Nano-Utils>=2.3',
        'pyyaml>=5.1',
        'numpy>=1.15',
        'scipy>=1.2',
        'pandas>=0.24',
        'schema>=0.7.1,!=0.7.5',
        'AssertionLib>=2.3',
        'noodles>=0.3.3',
        'h5py>=2.10',
        'qmflows>=0.11.0',
        'plams>=1.5.1',
    ],
    tests_require=tests_require,
    extras_require={
        'docs': docs_require,
        'test': tests_require,
        'test_no_optional': tests_require_no_optional,
    }
)
