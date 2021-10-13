"""Test the :mod:`sphinx` documentation generation."""

from __future__ import annotations

from os.path import join

import pytest
from nanoutils import delete_finally

try:
    from sphinx.application import Sphinx
except ImportError as ex:
    SPHINX_EX: None | ImportError = ex
else:
    SPHINX_EX = None

SRCDIR = CONFDIR = 'docs'
OUTDIR = join('tests', 'test_files', 'build')
DOCTREEDIR = join('tests', 'test_files', 'build', 'doctrees')


@delete_finally(OUTDIR)
@pytest.mark.skipif(SPHINX_EX is not None, reason="Requires Sphinx")
def test_sphinx_build() -> None:
    """Test :meth:`~sphinx.application.Sphinx.build`."""
    app = Sphinx(SRCDIR, CONFDIR, OUTDIR, DOCTREEDIR, buildername='html', warningiserror=True)
    app.build(force_all=True)
