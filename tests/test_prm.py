"""Tests for :class:`nanoCAT.ff.prm.PRMContainer`."""

import os
import pickle
from pathlib import Path
from tempfile import TemporaryFile
from itertools import zip_longest

import pandas as pd

from assertionlib import assertion
from nanoutils import delete_finally
from FOX import PRMContainer

PATH = Path('tests') / 'test_files'
PATH_PRM = PATH / 'Butoxide.prm'
PATH_TMP = PATH / 'tmp.prm'

PRM = PRMContainer.read(PATH_PRM)
PRM._pd_printoptions['display.max_rows'] = 10

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


def is_view(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """Check if all series in **df1** are views of their counterpart in **df2**."""
    iterator = ((v.values.base, df2[k].values.base) for k, v in df1.items())
    return all([ar1 is ar2 for ar1, ar2 in iterator])


def test_magic() -> None:
    """Tests for :class:`PRMContainer` magic methods."""
    # Test __eq__
    prm_shallow = PRM.copy(deep=False)
    prm_deep = PRM.copy(deep=True)
    assertion.eq(PRM, prm_shallow)
    assertion.eq(PRM, prm_deep)

    # Test copy, __deepcopy__ and __copy__
    attr_names = PRMContainer.__slots__[:-2]

    iterator1 = ((k, getattr(prm_deep, k), getattr(PRM, k)) for k in attr_names)
    for name, attr1, attr2 in iterator1:
        if isinstance(attr1, pd.DataFrame):
            assertion.assert_(is_view, attr1, attr2, invert=True, message=name)

    iterator2 = ((k, getattr(prm_shallow, k), getattr(PRM, k)) for k in attr_names)
    for name, attr1, attr2 in iterator2:
        if isinstance(attr1, pd.DataFrame):
            assertion.assert_(is_view, attr1, attr2, message=name)

    # Test __reduce__ and __setstate__
    dumps = pickle.dumps(PRM)
    loads = pickle.loads(dumps)
    assertion.eq(PRM, loads)
    assertion.eq(PRM._pd_printoptions, loads._pd_printoptions)
