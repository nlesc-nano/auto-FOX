"""Tests for :class:`nanoCAT.ff.prm.PRMContainer`."""

import os
from pathlib import Path
from tempfile import TemporaryFile
from itertools import zip_longest

from assertionlib import assertion
from nanoutils import delete_finally

from FOX import PRMContainer

PATH = Path('tests') / 'test_files'
PATH_PRM = PATH / 'Butoxide.prm'
PATH_TMP = PATH / 'tmp.prm'

PRM = PRMContainer.read(PATH_PRM)

if os.name == 'nt':
    STRIP = b'\r\n'
else:
    STRIP = b'\n'


@delete_finally(PATH_TMP)
def test_write() -> None:
    """Tests for :meth:`PSFContainer.write`."""
    PRM.write(PATH_TMP)
    with open(PATH_PRM) as f1, open(PATH_TMP) as f2:
        for i, j in zip_longest(f1, f2):
            assertion.eq(i, j)

    with open(PATH_PRM, 'rb') as f3, TemporaryFile() as f4:
        PRM.write(f4, bytes_encoding='utf-8')
        f4.seek(0)
        for i, j in zip_longest(f3, f4, fillvalue=b'<FILLVALUE>'):
            assertion.eq(i.rstrip(STRIP), j.rstrip(STRIP))
