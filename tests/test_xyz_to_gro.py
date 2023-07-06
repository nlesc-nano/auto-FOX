import pathlib

import pytest
import numpy as np
from FOX.recipes import xyz_to_gro, gro_to_xyz
from FOX import MultiMolecule, example_xyz


def test_xyz_to_gro(tmp_path: pathlib.Path) -> None:
    gro_file = tmp_path / "xyz_to_gro.gro"
    xyz_file = tmp_path / "gro_to_xyz.xyz"

    mol_ref = MultiMolecule.from_xyz(example_xyz)
    with pytest.warns(Warning, match="contains multiple molecules"):
        xyz_to_gro(mol_ref, gro_file)
    gro_to_xyz(gro_file, xyz_file)
    mol = MultiMolecule.from_xyz(xyz_file)

    np.testing.assert_allclose(mol[0], mol_ref[0], atol=1e-3, rtol=0)
    np.testing.assert_array_equal(mol_ref.symbol, mol.symbol)
