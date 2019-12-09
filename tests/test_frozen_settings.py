"""A module for testing the :class:`FOX.classes.frozen_settings.FrozenSettings` class."""

from scm.plams import Settings
from assertionlib import assertion

from FOX import FrozenSettings

_s = Settings()
_s.a.b.c.d = True
REF = FrozenSettings(_s)
HASH = hash(REF)


def test_missing():
    """Test :meth:`.FrozenSettings.__missing__`."""
    item = REF.b
    assertion.eq(item, FrozenSettings())
    assertion.contains(REF, 'b', invert=True)


def test_delitem():
    """Test :meth:`.FrozenSettings.__delitem__`."""
    assertion.assert_(REF.__delitem__, 'a', exception=TypeError)


def test_setitem():
    """Test :meth:`.FrozenSettings.__setitem__`."""
    assertion.assert_(REF.__setitem__, 'a', exception=TypeError)


def test_hash():
    """Test :meth:`.FrozenSettings.__hash__`."""
    copy = REF.copy()
    assertion.eq(hash(copy), HASH)
    assertion.eq(HASH, copy._hash)


def test_copy():
    """Test :meth:`.FrozenSettings.__copy__`."""
    copy = REF.copy()
    assertion.is_not(copy, REF)
    assertion.eq(copy, REF)


def test_setnested():
    """Test :meth:`.FrozenSettings.set_nested`."""
    key_tuple = ('b', 'c', 'd')
    value = True
    assertion.assert_(REF.set_nested, key_tuple, value, exception=TypeError)


def test_flatten():
    """Test :meth:`.FrozenSettings.flatten`."""
    ref_flat = REF.flatten()
    assertion.is_(ref_flat[('a', 'b', 'c', 'd')], True)


def test_unflatten():
    """Test :meth:`.FrozenSettings.unflatten`."""
    ref_flat = REF.flatten()
    ref_unflat = ref_flat.unflatten()
    assertion.eq(ref_unflat, REF)
