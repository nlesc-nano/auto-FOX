"""A module with yaml-related tests."""

import yaml
from yaml.constructor import ConstructorError

from FOX.yaml import UniqueLoader
from assertionlib import assertion

YAML1 = 'a: True'
YAML2 = 'a: False\na: True'


def test_yaml() -> None:
    """Tests for :class:`FOX.yaml.UniqueLoader`."""
    dct1 = yaml.load(YAML1, Loader=yaml.Loader)
    dct2 = yaml.load(YAML1, Loader=UniqueLoader)
    dct3 = yaml.load(YAML2, Loader=yaml.Loader)

    assertion.eq(dct1, dct2)
    assertion.eq(dct1, dct3)

    assertion.assert_(yaml.load, YAML2, Loader=UniqueLoader, exception=ConstructorError)
