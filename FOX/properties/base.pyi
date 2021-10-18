import types
from typing import Generic, Callable, Any, ClassVar, TypeVar, overload, Tuple, Type
from typing_extensions import Literal, Protocol, TypedDict
from collections.abc import MutableMapping

import numpy as np
from numpy.typing import NDArray
from qmflows.packages.cp2k_package import CP2K_Result

__all__ = ['FromResult', 'get_attr', 'call_method']

T1 = TypeVar("T1")
T2 = TypeVar("T2")
FT = TypeVar("FT", bound=Callable[..., Any])
SCT_co = TypeVar("SCT_co", covariant=True, bound=np.generic)

class ResultFunc(Protocol):
    def __call__(
        __self,
        self: FromResult[Callable[..., T1]],
        result: CP2K_Result,
        *,
        reduce: None | str | Callable[[T1], Any] = ...,
        axis: None | int | Tuple[int, ...] = ...,
        return_unit: str = ...,
        **kwargs: Any,
    ) -> Any: ...

class ReduceFunc(Protocol[SCT_co]):
    @overload
    def __call__(self, __a: np.float64, *, axis: None | int | Tuple[int, ...] = ...) -> SCT_co: ...
    @overload
    def __call__(self, __a: NDArray[np.float64], *, axis: None = ...) -> SCT_co: ...
    @overload
    def __call__(self, __a: NDArray[np.float64], *, axis: int | Tuple[int, ...]) -> Any: ...

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

class FromResult(Generic[FT]):
    REDUCTION_NAMES: ClassVar[ReductionDict]
    def __init__(self, func: FT, result_func: None | ResultFunc = None) -> None: ...
    @property
    def __name__(self) -> str: ...
    @property
    def __qualname__(self) -> str: ...
    @property
    def __module__(self) -> str: ...  # type: ignore[override]
    @property
    def __doc__(self) -> None | str: ...  # type: ignore[override]
    @property
    def __annotations__(self) -> dict[str, Any]: ...  # type: ignore[override]
    @property
    def __text_signature__(self) -> None | str: ...
    @property
    def __closure__(self) -> None | Tuple[types._Cell, ...]: ...
    @property
    def __defaults__(self) -> None | Tuple[Any, ...]: ...
    @property
    def __globals__(self) -> dict[str, Any]: ...
    @property
    def __kwdefaults__(self) -> dict[str, Any]: ...
    @property
    def __code__(self) -> types.CodeType: ...
    @property
    def __call__(self) -> FT: ...
    def __get__(self, obj: None | object, type: None | type) -> types.MethodType: ...
    def __hash__(self) -> int: ...
    def __eq__(self, value: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __reduce__(self) -> str: ...
    def __copy__(self: T1) -> T1: ...
    def __deepcopy__(self: T1, memo: None | dict[int, Any] = None) -> T1: ...
    def __dir__(self) -> list[str]: ...
    def _set_result_func(self: T1, result_func: ResultFunc) -> T1: ...

    @overload
    def from_result(self: FromResult[Callable[..., T1]], result: CP2K_Result, *, reduce: None = ..., axis: None = ..., return_unit: str = ...) -> T1: ...
    @overload
    def from_result(self: FromResult[Callable[..., T1]], result: CP2K_Result, *, reduce: Callable[[T1], T2], axis: None = ..., return_unit: str = ...) -> T2: ...
    @overload
    def from_result(self, result: CP2K_Result, *, reduce: Float64Names, axis: None = ..., return_unit: str = ...) -> np.float64: ...
    @overload
    def from_result(self, result: CP2K_Result, *, reduce: IntPNames, axis: None = ..., return_unit: str = ...) -> np.intp: ...
    @overload
    def from_result(self, result: CP2K_Result, *, reduce: BoolNames, axis: None = ..., return_unit: str = ...) -> np.bool_: ...
    @overload
    def from_result(self, result: CP2K_Result, *, reduce: Float64Names | IntPNames | BoolNames, axis: int | Tuple[int, ...], return_unit: str = ...) -> Any: ...

    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: None = ..., axis: None = ...) -> T1: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: Callable[[T1], T2], axis: None = ...) -> T2: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: Float64Names, axis: None = ...) -> np.float64: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: IntPNames, axis: None = ...) -> np.intp: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: BoolNames, axis: None = ...) -> np.bool_: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: Float64Names | IntPNames | BoolNames, axis: int | Tuple[int, ...]) -> Any: ...

    @staticmethod
    def _pop(dct: MutableMapping[T1, T2], key: T1, callback: Callable[[], T2]) -> T2: ...

@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: None = ..., axis: None = ...) -> Any: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: Callable[[Any], T2], axis: None = ...) -> T2: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: Float64Names, axis: None = ...) -> np.float64: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: IntPNames, axis: None = ...) -> np.intp: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: BoolNames, axis: None = ...) -> np.bool_: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: Float64Names | IntPNames | BoolNames, axis: int | Tuple[int, ...]) -> Any: ...

@overload
def call_method(obj: object, name: str, *args: Any, reduce: None = ..., axis: None = ..., **kwargs: Any) -> Any: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: Callable[[Any], T2], axis: None = ..., **kwargs: Any) -> T2: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: Float64Names, axis: None = ..., **kwargs: Any) -> np.float64: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: IntPNames, axis: None = ..., **kwargs: Any) -> np.intp: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: BoolNames, axis: None = ..., **kwargs: Any) -> np.bool_: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: Float64Names | IntPNames | BoolNames, axis: int | Tuple[int, ...], **kwargs: Any) -> Any: ...
