import sys
from typing import Callable

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 8):
    from typing import TypedDict, Literal
else:
    from typing_extensions import TypedDict, Literal

Float64Names = Literal[
    "min",
    "max",
    "mean",
    "sum",
    "product",
    "var",
    "std",
    "ptp",
    "norm",
    "add",
    "arctan2",
    "copysign",
    "true_divide",
    "floor_divide",
    "float_power",
    "fmax",
    "fmin",
    "fmod",
    "heaviside",
    "hypot",
    "ldexp",
    "logaddexp",
    "logaddexp2",
    "maximum",
    "minimum",
    "remainder",
    "multiply",
    "nextafter",
    "power",
    "subtract",
    "agm",
    "beta",
    "betaln",
    "binom",
    "boxcox",
    "boxcox1p",
    "chdtr",
    "chdtrc",
    "chdtri",
    "chdtriv",
    "ellipeinc",
    "ellipkinc",
    "eval_chebyc",
    "eval_chebys",
    "eval_chebyt",
    "eval_chebyu",
    "eval_hermite",
    "eval_hermitenorm",
    "eval_laguerre",
    "eval_legendre",
    "eval_sh_chebyt",
    "eval_sh_chebyu",
    "eval_sh_legendre",
    "expn",
    "gammainc",
    "gammaincc",
    "gammainccinv",
    "gammaincinv",
    "hankel1",
    "hankel1e",
    "hankel2",
    "hankel2e",
    "huber",
    "hyp0f1",
    "inv_boxcox",
    "inv_boxcox1p",
    "iv",
    "ive",
    "jv",
    "jve",
    "kl_div",
    "kn",
    "kv",
    "kve",
    "mathieu_a",
    "mathieu_b",
    "modstruve",
    "owens_t",
    "pdtr",
    "pdtrc",
    "pdtri",
    "pdtrik",
    "poch",
    "pseudo_huber",
    "rel_entr",
    "smirnov",
    "smirnovi",
    "stdtr",
    "stdtridf",
    "stdtrit",
    "struve",
    "tklmbda",
    "xlog1py",
    "xlogy",
    "yn",
    "yv",
    "yve",
]
IntPNames = Literal[
    "argmin",
    "argmax",
]
BoolNames = Literal[
    "all",
    "any",
    "equal",
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "not_equal",
]

class ReductionDict(TypedDict):
    min: Callable[[npt.ArrayLike], np.float64]
    max: Callable[[npt.ArrayLike], np.float64]
    mean: Callable[[npt.ArrayLike], np.float64]
    sum: Callable[[npt.ArrayLike], np.float64]
    product: Callable[[npt.ArrayLike], np.float64]
    var: Callable[[npt.ArrayLike], np.float64]
    std: Callable[[npt.ArrayLike], np.float64]
    ptp: Callable[[npt.ArrayLike], np.float64]
    norm: Callable[[npt.ArrayLike], np.float64]
    argmin: Callable[[npt.ArrayLike], np.intp]
    argmax: Callable[[npt.ArrayLike], np.intp]
    all: Callable[[npt.ArrayLike], np.bool_]
    any: Callable[[npt.ArrayLike], np.bool_]

    # numpy ufuncs
    add: Callable[[npt.ArrayLike], np.float64]
    arctan2: Callable[[npt.ArrayLike], np.float64]
    copysign: Callable[[npt.ArrayLike], np.float64]
    true_divide: Callable[[npt.ArrayLike], np.float64]
    floor_divide: Callable[[npt.ArrayLike], np.float64]
    float_power: Callable[[npt.ArrayLike], np.float64]
    fmax: Callable[[npt.ArrayLike], np.float64]
    fmin: Callable[[npt.ArrayLike], np.float64]
    fmod: Callable[[npt.ArrayLike], np.float64]
    heaviside: Callable[[npt.ArrayLike], np.float64]
    hypot: Callable[[npt.ArrayLike], np.float64]
    ldexp: Callable[[npt.ArrayLike], np.float64]
    logaddexp: Callable[[npt.ArrayLike], np.float64]
    logaddexp2: Callable[[npt.ArrayLike], np.float64]
    maximum: Callable[[npt.ArrayLike], np.float64]
    minimum: Callable[[npt.ArrayLike], np.float64]
    remainder: Callable[[npt.ArrayLike], np.float64]
    multiply: Callable[[npt.ArrayLike], np.float64]
    nextafter: Callable[[npt.ArrayLike], np.float64]
    power: Callable[[npt.ArrayLike], np.float64]
    subtract: Callable[[npt.ArrayLike], np.float64]

    equal: Callable[[npt.ArrayLike], np.bool_]
    greater: Callable[[npt.ArrayLike], np.bool_]
    greater_equal: Callable[[npt.ArrayLike], np.bool_]
    less: Callable[[npt.ArrayLike], np.bool_]
    less_equal: Callable[[npt.ArrayLike], np.bool_]
    logical_and: Callable[[npt.ArrayLike], np.bool_]
    logical_or: Callable[[npt.ArrayLike], np.bool_]
    logical_xor: Callable[[npt.ArrayLike], np.bool_]
    not_equal: Callable[[npt.ArrayLike], np.bool_]

    # scipy.special ufuncs
    agm: Callable[[npt.ArrayLike], np.float64]
    beta: Callable[[npt.ArrayLike], np.float64]
    betaln: Callable[[npt.ArrayLike], np.float64]
    binom: Callable[[npt.ArrayLike], np.float64]
    boxcox: Callable[[npt.ArrayLike], np.float64]
    boxcox1p: Callable[[npt.ArrayLike], np.float64]
    chdtr: Callable[[npt.ArrayLike], np.float64]
    chdtrc: Callable[[npt.ArrayLike], np.float64]
    chdtri: Callable[[npt.ArrayLike], np.float64]
    chdtriv: Callable[[npt.ArrayLike], np.float64]
    ellipeinc: Callable[[npt.ArrayLike], np.float64]
    ellipkinc: Callable[[npt.ArrayLike], np.float64]
    eval_chebyc: Callable[[npt.ArrayLike], np.float64]
    eval_chebys: Callable[[npt.ArrayLike], np.float64]
    eval_chebyt: Callable[[npt.ArrayLike], np.float64]
    eval_chebyu: Callable[[npt.ArrayLike], np.float64]
    eval_hermite: Callable[[npt.ArrayLike], np.float64]
    eval_hermitenorm: Callable[[npt.ArrayLike], np.float64]
    eval_laguerre: Callable[[npt.ArrayLike], np.float64]
    eval_legendre: Callable[[npt.ArrayLike], np.float64]
    eval_sh_chebyt: Callable[[npt.ArrayLike], np.float64]
    eval_sh_chebyu: Callable[[npt.ArrayLike], np.float64]
    eval_sh_legendre: Callable[[npt.ArrayLike], np.float64]
    expn: Callable[[npt.ArrayLike], np.float64]
    gammainc: Callable[[npt.ArrayLike], np.float64]
    gammaincc: Callable[[npt.ArrayLike], np.float64]
    gammainccinv: Callable[[npt.ArrayLike], np.float64]
    gammaincinv: Callable[[npt.ArrayLike], np.float64]
    hankel1: Callable[[npt.ArrayLike], np.float64]
    hankel1e: Callable[[npt.ArrayLike], np.float64]
    hankel2: Callable[[npt.ArrayLike], np.float64]
    hankel2e: Callable[[npt.ArrayLike], np.float64]
    huber: Callable[[npt.ArrayLike], np.float64]
    hyp0f1: Callable[[npt.ArrayLike], np.float64]
    inv_boxcox: Callable[[npt.ArrayLike], np.float64]
    inv_boxcox1p: Callable[[npt.ArrayLike], np.float64]
    iv: Callable[[npt.ArrayLike], np.float64]
    ive: Callable[[npt.ArrayLike], np.float64]
    jv: Callable[[npt.ArrayLike], np.float64]
    jve: Callable[[npt.ArrayLike], np.float64]
    kl_div: Callable[[npt.ArrayLike], np.float64]
    kn: Callable[[npt.ArrayLike], np.float64]
    kv: Callable[[npt.ArrayLike], np.float64]
    kve: Callable[[npt.ArrayLike], np.float64]
    mathieu_a: Callable[[npt.ArrayLike], np.float64]
    mathieu_b: Callable[[npt.ArrayLike], np.float64]
    modstruve: Callable[[npt.ArrayLike], np.float64]
    owens_t: Callable[[npt.ArrayLike], np.float64]
    pdtr: Callable[[npt.ArrayLike], np.float64]
    pdtrc: Callable[[npt.ArrayLike], np.float64]
    pdtri: Callable[[npt.ArrayLike], np.float64]
    pdtrik: Callable[[npt.ArrayLike], np.float64]
    poch: Callable[[npt.ArrayLike], np.float64]
    pseudo_huber: Callable[[npt.ArrayLike], np.float64]
    rel_entr: Callable[[npt.ArrayLike], np.float64]
    smirnov: Callable[[npt.ArrayLike], np.float64]
    smirnovi: Callable[[npt.ArrayLike], np.float64]
    stdtr: Callable[[npt.ArrayLike], np.float64]
    stdtridf: Callable[[npt.ArrayLike], np.float64]
    stdtrit: Callable[[npt.ArrayLike], np.float64]
    struve: Callable[[npt.ArrayLike], np.float64]
    tklmbda: Callable[[npt.ArrayLike], np.float64]
    xlog1py: Callable[[npt.ArrayLike], np.float64]
    xlogy: Callable[[npt.ArrayLike], np.float64]
    yn: Callable[[npt.ArrayLike], np.float64]
    yv: Callable[[npt.ArrayLike], np.float64]
    yve: Callable[[npt.ArrayLike], np.float64]
