from typing import Sequence, Union, Iterable
from itertools import chain

import numpy as np
import pandas as pd

from FOX import ARMC, PSFContainer, MultiMolecule
from FOX.ff.lj_param import estimate_lj
from FOX.ff.lj_uff import UFF_DF
from FOX.ff.lj_dataframe import LJDataFrame


def start(armc: ARMC, mode: str = 'rdf',
          columns: Union[str, Iterable[str]] = ('epsilon', 'sigma')) -> pd.DataFrame:
    # Parse and sort the columns
    columns = sorted(columns) if not isinstance(columns, str) else [columns]
    param_df = _pre_process(armc)
    if mode == 'rdf':
        df = rdf(param_df, armc.molecule, columns=columns)
    elif mode == 'uff':
        df = uff(param_df, armc.molecule, columns=columns)
    return df


def _pre_process(armc: ARMC) -> LJDataFrame:
    # Update all atom based on the .psf file
    mol_tup = armc.molecule
    for mol, s in zip(mol_tup, armc.md_settings):
        psf = PSFContainer.read(s.input.force_eval.subsys.topology.conn_file_name)
        mol.atoms = psf.to_atom_dict()

    # COnstruct a dataframe with all to-be guessed parameters
    df = LJDataFrame(np.nan, index=chain.from_iterable(mol.atoms.keys() for mol in mol_tup))
    for s in armc.md_settings:
        df.overlay_prm(s.input.force_eval.mm.forcefield.parm_file_name)
        df.overlay_cp2k_settings(s)
    del df['charge']
    return df[df.isna().any(axis=1)]


def uff(df: pd.DataFrame, mol_list: Iterable[MultiMolecule],
        columns: Sequence[str] = ['epsilon', 'sigma']) -> pd.DataFrame:
    pass


def rdf(df: pd.DataFrame, mol_list: Iterable[MultiMolecule],
        columns: Sequence[str] = ['epsilon', 'sigma']) -> pd.DataFrame:

    # Construct the RDF and guess the parameters
    rdf_gen = (mol.init_rdf() for mol in mol_list)
    for rdf in rdf_gen:
        guess = estimate_lj(rdf)
        guess.index = pd.MultiIndex.from_tuples(sorted(i.split()) for i in guess.index)
        df.update(guess[columns], overwrite=False)
    return df
