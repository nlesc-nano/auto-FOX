"""A module for computing Debye scattering factors.

Index
-----
.. currentmodule:: FOX.functions.debye
.. autosummary::
    get_debye_scattering
    SCATERING_FACTORS

API
---
.. autofunction:: get_debye_scattering
.. autodata:: SCATERING_FACTORS

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from numpy import str_ as U, float64 as f8
    from typing_extensions import TypedDict

    class _CoefDict(TypedDict):
        a: list[float]
        b: list[float]
        c: list[float]

__all__ = ["SCATERING_FACTORS", "get_debye_scattering"]


def _load_scattering_df() -> pd.DataFrame:
    """Load the dataframe with scattering coefficients.

    Coefficients are taken from:
    International Tables for Crystallography (2006). Vol. C, ch. 6.1, pp. 578-580, table 6.1.1.4

    .. code-block:: python

        >>> df = _load_scattering_df()
        >>> print(df)
        coef            a                                         b                                         c
        i               0          1          2        3          0         1          2         3          0
        symbol
        H        0.493002   0.322912   0.140191  0.04081  10.510900  26.12570    3.14236   57.7997   0.003038
        He       0.873400   0.630900   0.311200  0.17800   9.103700   3.35680   22.92760    0.9821   0.006400
        Li       1.128200   0.750800   0.617500  0.46530   3.954600   1.05240   85.39050  168.2610   0.037700
        Be       1.591900   1.127800   0.539100  0.70290  43.642700   1.86230  103.48300    0.5420   0.038500
        B        2.054500   1.332600   1.097900  0.70680  23.218500   1.02100   60.34980    0.1403  -0.193200
        ...           ...        ...        ...      ...        ...       ...        ...       ...        ...
        Pu      36.525400  23.808300  16.770700  3.47947   0.499384   3.26371   14.94550  105.9800  13.381200
        Am      36.670600  24.099200  17.341500  3.49331   0.483629   3.20647   14.31360  102.2730  13.359200
        Cm      36.648800  24.409600  17.399000  4.21665   0.465154   3.08997   13.43460   88.4834  13.288700
        Bk      36.788100  24.773600  17.891900  4.23284   0.451018   3.04619   12.89460   86.0030  13.275400
        Cf      36.918500  25.199500  18.331700  4.24391   0.437533   3.00775   12.40440   83.7881  13.267400

    """  # noqa: E501
    root = Path(__file__).parents[1]
    with open(root / "data" / "scattering.yaml", "r") as f:
        dct: dict[str, _CoefDict] = yaml.load(f, Loader=yaml.SafeLoader)

    columns = pd.MultiIndex.from_tuples([
        ("a", 0),
        ("a", 1),
        ("a", 2),
        ("a", 3),
        ("b", 0),
        ("b", 1),
        ("b", 2),
        ("b", 3),
        ("c", 0),
    ], names=("Coefficients", "i"))

    index = []
    data = np.empty((len(dct), 9), order="F", dtype=np.float64)
    for i, (k, v) in enumerate(dct.items()):
        index.append(k)
        data[i] = v["a"] + v["b"] + v["c"]
    return pd.DataFrame(data, index=pd.Index(index, name="symbol"), columns=columns)


#: A dataframe with generalized X-ray scattering coefficients.
#:
#: See Also
#: --------
#: International Tables for Crystallography (2006). Vol. C, ch. 6.1, pp. 578-580, table 6.1.1.4
SCATERING_FACTORS = _load_scattering_df()


def _get_scattering(symbol: NDArray[U], q: NDArray[f8]) -> NDArray[f8]:
    """Computer the scattering factors for the given atomic symbols."""
    stol2 = (q / (4 * np.pi))**2
    a = SCATERING_FACTORS.loc[symbol, "a"].values
    b = SCATERING_FACTORS.loc[symbol, "b"].values
    ret = SCATERING_FACTORS.loc[symbol, ("c", 0)].values.copy()
    ret += (a * np.exp(-b * stol2)).sum(axis=1)
    return ret


@np.errstate(invalid="ignore")
def get_debye_scattering(
    dist_mat: NDArray[f8],
    symbols1: NDArray[U],
    symbols2: NDArray[U],
    scattering_vector: NDArray[f8],
    validate_param: bool = True,
) -> NDArray[f8]:
    """Placeholder."""
    if validate_param:
        dist_mat = np.array(dist_mat, dtype=np.float64, ndmin=3, copy=False)
        scattering_vector = np.array(scattering_vector, dtype=np.float64, ndmin=1, copy=False)
        symbols1 = np.array(symbols1, dtype=np.str_, ndmin=1, copy=False)
        symbols2 = np.array(symbols2, dtype=np.str_, ndmin=1, copy=False)

        try:
            assert dist_mat.ndim == 3, "Invalid `dist_mat` dimensionality"
            assert symbols1.ndim == 1, "Invalid `symbols1` dimensionality"
            assert symbols2.ndim == 1, "Invalid `symbols2` dimensionality"
            assert scattering_vector.ndim == 1, "Invalid `scattering_vector` dimensionality"
        except AssertionError as ex:
            raise ValueError(str(ex)) from None

    q_r_ij = scattering_vector * dist_mat
    f_ij = (
        _get_scattering(symbols1, scattering_vector)[..., None] *
        _get_scattering(symbols2, scattering_vector)[None, ...]
    )

    ret = np.sin(q_r_ij)
    ret /= q_r_ij
    ret *= f_ij[None, ...]
    return np.nansum(ret, axis=(1, 2))
