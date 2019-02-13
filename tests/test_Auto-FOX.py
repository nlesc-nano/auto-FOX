#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the Auto-FOX module.
"""
import pytest

from Auto-FOX import Auto-FOX


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise(ValueError)


# Fixture example
@pytest.fixture
def an_object():
    return {}


def test_Auto-FOX(an_object):
    assert an_object == {}
