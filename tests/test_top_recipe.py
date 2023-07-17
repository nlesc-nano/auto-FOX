from __future__ import annotations

from pathlib import Path

from FOX import TOPContainer
from FOX.recipes import create_top
from assertionlib import assertion

PATH = Path('tests') / 'test_files'


def test_create_top() -> None:
    top = create_top(
        mol_count=[5],
        rtf_files=[PATH / "ola.rtf"],
        prm_files=[PATH / "ola.prm"],
    )
    ref = TOPContainer.from_file(PATH / "top_recipe_ref.top")
    assertion.assert_(top.allclose, ref, rtol=0, atol=0.0001)
