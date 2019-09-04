"""A module for testing the :class:`FOX.classes.frozen_settings.FrozenSettings` class."""

from scm.plams import Settings

from FOX.classes.frozen_settings import FrozenSettings

_s = Settings()
_s.a.b.c.d = True
REF = FrozenSettings(_s)
HASH = hash(REF)


def test_missing():
    """Test :meth:`.FrozenSettings.__missing__`."""
    item = REF.b
    assert item == FrozenSettings()
    assert 'b' not in REF


def test_delitem():
    """Test :meth:`.FrozenSettings.__delitem__`."""
    try:
        del REF['a']
    except TypeError:
        pass
    else:
        raise AssertionError


def test_setitem():
    """Test :meth:`.FrozenSettings.__setitem__`."""
    try:
        REF['b'] = True
    except TypeError:
        pass
    else:
        raise AssertionError


def test_hash():
    """Test :meth:`.FrozenSettings.__hash__`."""
    copy = REF.copy()
    assert hash(copy) == HASH
    assert HASH == copy._hash


def test_copy():
    """Test :meth:`.FrozenSettings.__copy__`."""
    copy = REF.copy()
    assert id(copy) != id(REF)
    assert copy == REF


def test_setnested():
    """Test :meth:`.FrozenSettings.set_nested`."""
    key_tuple = ('b', 'c', 'd')
    value = True
    try:
        REF.set_nested(key_tuple, value)
    except TypeError:
        pass
    else:
        raise AssertionError


def test_flatten():
    """Test :meth:`.FrozenSettings.flatten`."""
    ref_flat = REF.flatten()
    assert ref_flat[('a', 'b', 'c', 'd')] is True


def test_unflatten():
    """Test :meth:`.FrozenSettings.unflatten`."""
    ref_flat = REF.flatten()
    ref_unflat = ref_flat.unflatten()
    assert ref_unflat == REF
