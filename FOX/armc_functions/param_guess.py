from types import MappingProxyType
from typing import Iterable
from functools import partial
from itertools import combinations_with_replacement

import pandas as pd

from FOX.ff.lj_uff import combine_sigma, combine_epsilon
from FOX.ff.lj_calculate import LJDataFrame


SIGMA_MAPPING = MappingProxyType({
    'uff': combine_sigma,
    'rdf': ...
})

EPSILON_MAPPING = MappingProxyType({
    'uff': combine_epsilon,
    'rdf': ...
})


def _parse_uff(atoms: Iterable[str]):
    iterator = combinations_with_replacement(sorted(atoms), r=2)
    return pd.DataFrame({
        ij: {'sigma': combine_sigma(*ij), 'epsilon': combine_epsilon(*ij)} for ij in iterator
    }).T


def guesstimate(atoms: Iterable[str]):
    uff_df = _parse_uff(atoms)
