from __future__ import annotations

from pathlib import Path

from FOX import TOPContainer
from FOX.recipes import create_top
from assertionlib import assertion

PATH = Path('tests') / 'test_files'


def test_create_top(tmp_path: Path) -> None:
    top = create_top(
        mol_count=[5],
        rtf_files=[PATH / "ola.rtf"],
        prm_files=[PATH / "ola.prm"],
    )
    top.to_file(tmp_path / "test_create_top.top")

    top2 = TOPContainer.from_file(tmp_path / "test_create_top.top")
    ref = TOPContainer.from_file(PATH / "top_recipe_ref.top")
    assertion.eq(top2, ref)
