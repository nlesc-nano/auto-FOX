"""Tests for :class:`nanoCAT.ff.prm.PRMContainer`."""

import os
from os.path import join
from tempfile import TemporaryFile
from itertools import zip_longest

from assertionlib import assertion

from FOX import PRMContainer

PATH: str = join('tests', 'test_files')
PRM: PRMContainer = PRMContainer.read(join(PATH, 'Butoxide.prm'))


def test_write() -> None:
    """Tests for :meth:`PSFContainer.write`."""
    filename1 = join(PATH, 'Butoxide.prm')
    filename2 = join(PATH, 'tmp.prm')

    try:
        PRM.write(filename2)
        with open(filename1) as f1, open(filename2) as f2:
            for i, j in zip_longest(f1, f2):
                assertion.eq(i, j)
    finally:
        if os.path.isfile(filename2):
            os.remove(filename2)

    with open(filename1, 'rb') as f1, TemporaryFile() as f2:
        PRM.write(f2, encoding='utf-8')
        f2.seek(0)
        for i, j in zip_longest(f1, f2):
            assertion.eq(i, j)
