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
    'sphinx>=2.1',
    'sphinx_rtd_theme',
    'matplotlib'
]

tests_require = [
    'pytest>=5.4.0',
    'pytest-cov',
    'pyflakes>=2.1.1',
    'pytest-flake8>=1.0.5',
    'pytest-pydocstyle>=2.1',
    'auto-FOX-data@git+https://github.com/nlesc-nano/auto-FOX-data@1.1.5',
]
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
        'FOX.recipes'
    ],
    package_dir={'FOX': 'FOX'},
    package_data={'FOX': [
        'data/*.xyz',
        'data/*.yaml',
        'data/*.csv',
        'py.typed',
        '*.pyi'
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
        'python-3',
        'python-3.7',
        'python-3.8'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Software Development :: Libraries ',
        'Typing :: Typed'
    ],
    python_requires='>=3.7',
    install_requires=[
        'Nano-Utils>=1.2',
        'pyyaml>=5.1',
        'numpy>=1.15',
        'scipy>=1.2',
        'pandas',
        'schema>=0.7.1',
        'AssertionLib>=2.3',
        'noodles>=0.3.3',
        'h5py>=2.10',
        'qmflows@git+https://github.com/SCM-NV/qmflows@master',
        'plams@git+https://github.com/SCM-NV/PLAMS@master',
    ],
    setup_requires=[
        'pytest-runner'
    ],
    tests_require=tests_require,
    extras_require={'docs': docs_require, 'test': tests_require}
)
