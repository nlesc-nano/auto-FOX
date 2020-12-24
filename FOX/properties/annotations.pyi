import sys
from typing import Callable, overload, TypeVar, Union, Sequence, Any, Optional

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 8):
    from typing import TypedDict, Literal, Protocol
else:
    from typing_extensions import TypedDict, Literal, Protocol

__all__ = [
    "ShapeLike",
    "ScalarOrArray",
    "ReduceFunc",
    "Float64Names",
    "IntPNames",
    "BoolNames",
    "ReductionDict",
    "NDArray",
]

SCT = TypeVar("SCT", bound=np.generic)
SCT_co = TypeVar("SCT_co", covariant=True, bound=np.generic)

ShapeLike = Union[int, Sequence[int]]
NDArray = np.ndarray[Any, np.dtype[SCT]]
ScalarOrArray = Union[SCT, NDArray[SCT]]

class ReduceFunc(Protocol[SCT_co]):
    @overload
    def __call__(self, a: np.float64, *, axis: Optional[ShapeLike] = ...) -> SCT_co: ...
    @overload
    def __call__(self, a: NDArray[np.float64], *, axis: None = ...) -> ScalarOrArray[SCT_co]: ...
    @overload
    def __call__(self, a: NDArray[np.float64], *, axis: ShapeLike) -> ScalarOrArray[SCT_co]: ...

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
    min: ReduceFunc[np.float64]
    max: ReduceFunc[np.float64]
    mean: ReduceFunc[np.float64]
    sum: ReduceFunc[np.float64]
    product: ReduceFunc[np.float64]
    var: ReduceFunc[np.float64]
    std: ReduceFunc[np.float64]
    ptp: ReduceFunc[np.float64]
    norm: ReduceFunc[np.float64]
    argmin: ReduceFunc[np.intp]
    argmax: ReduceFunc[np.intp]
    all: ReduceFunc[np.bool_]
    any: ReduceFunc[np.bool_]

    # numpy ufuncs
    add: ReduceFunc[np.float64]
    arctan2: ReduceFunc[np.float64]
    copysign: ReduceFunc[np.float64]
    true_divide: ReduceFunc[np.float64]
    floor_divide: ReduceFunc[np.float64]
    float_power: ReduceFunc[np.float64]
    fmax: ReduceFunc[np.float64]
    fmin: ReduceFunc[np.float64]
    fmod: ReduceFunc[np.float64]
    heaviside: ReduceFunc[np.float64]
    hypot: ReduceFunc[np.float64]
    ldexp: ReduceFunc[np.float64]
    logaddexp: ReduceFunc[np.float64]
    logaddexp2: ReduceFunc[np.float64]
    maximum: ReduceFunc[np.float64]
    minimum: ReduceFunc[np.float64]
    remainder: ReduceFunc[np.float64]
    multiply: ReduceFunc[np.float64]
    nextafter: ReduceFunc[np.float64]
    power: ReduceFunc[np.float64]
    subtract: ReduceFunc[np.float64]

    equal: ReduceFunc[np.bool_]
    greater: ReduceFunc[np.bool_]
    greater_equal: ReduceFunc[np.bool_]
    less: ReduceFunc[np.bool_]
    less_equal: ReduceFunc[np.bool_]
    logical_and: ReduceFunc[np.bool_]
    logical_or: ReduceFunc[np.bool_]
    logical_xor: ReduceFunc[np.bool_]
    not_equal: ReduceFunc[np.bool_]

    # scipy.special ufuncs
    agm: ReduceFunc[np.float64]
    beta: ReduceFunc[np.float64]
    betaln: ReduceFunc[np.float64]
    binom: ReduceFunc[np.float64]
    boxcox: ReduceFunc[np.float64]
    boxcox1p: ReduceFunc[np.float64]
    chdtr: ReduceFunc[np.float64]
    chdtrc: ReduceFunc[np.float64]
    chdtri: ReduceFunc[np.float64]
    chdtriv: ReduceFunc[np.float64]
    ellipeinc: ReduceFunc[np.float64]
    ellipkinc: ReduceFunc[np.float64]
    eval_chebyc: ReduceFunc[np.float64]
    eval_chebys: ReduceFunc[np.float64]
    eval_chebyt: ReduceFunc[np.float64]
    eval_chebyu: ReduceFunc[np.float64]
    eval_hermite: ReduceFunc[np.float64]
    eval_hermitenorm: ReduceFunc[np.float64]
    eval_laguerre: ReduceFunc[np.float64]
    eval_legendre: ReduceFunc[np.float64]
    eval_sh_chebyt: ReduceFunc[np.float64]
    eval_sh_chebyu: ReduceFunc[np.float64]
    eval_sh_legendre: ReduceFunc[np.float64]
    expn: ReduceFunc[np.float64]
    gammainc: ReduceFunc[np.float64]
    gammaincc: ReduceFunc[np.float64]
    gammainccinv: ReduceFunc[np.float64]
    gammaincinv: ReduceFunc[np.float64]
    hankel1: ReduceFunc[np.float64]
    hankel1e: ReduceFunc[np.float64]
    hankel2: ReduceFunc[np.float64]
    hankel2e: ReduceFunc[np.float64]
    huber: ReduceFunc[np.float64]
    hyp0f1: ReduceFunc[np.float64]
    inv_boxcox: ReduceFunc[np.float64]
    inv_boxcox1p: ReduceFunc[np.float64]
    iv: ReduceFunc[np.float64]
    ive: ReduceFunc[np.float64]
    jv: ReduceFunc[np.float64]
    jve: ReduceFunc[np.float64]
    kl_div: ReduceFunc[np.float64]
    kn: ReduceFunc[np.float64]
    kv: ReduceFunc[np.float64]
    kve: ReduceFunc[np.float64]
    mathieu_a: ReduceFunc[np.float64]
    mathieu_b: ReduceFunc[np.float64]
    modstruve: ReduceFunc[np.float64]
    owens_t: ReduceFunc[np.float64]
    pdtr: ReduceFunc[np.float64]
    pdtrc: ReduceFunc[np.float64]
    pdtri: ReduceFunc[np.float64]
    pdtrik: ReduceFunc[np.float64]
    poch: ReduceFunc[np.float64]
    pseudo_huber: ReduceFunc[np.float64]
    rel_entr: ReduceFunc[np.float64]
    smirnov: ReduceFunc[np.float64]
    smirnovi: ReduceFunc[np.float64]
    stdtr: ReduceFunc[np.float64]
    stdtridf: ReduceFunc[np.float64]
    stdtrit: ReduceFunc[np.float64]
    struve: ReduceFunc[np.float64]
    tklmbda: ReduceFunc[np.float64]
    xlog1py: ReduceFunc[np.float64]
    xlogy: ReduceFunc[np.float64]
    yn: ReduceFunc[np.float64]
    yv: ReduceFunc[np.float64]
    yve: ReduceFunc[np.float64]
