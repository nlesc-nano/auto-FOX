import types
import inspect
from abc import ABCMeta, abstractmethod
from typing import Generic, Callable, Any, ClassVar, TypeVar, overload, Optional, MutableMapping, Tuple, Type

from qmflows.packages import Result
import numpy as np
import numpy.typing as npt

from FOX.properties.annotations import ReductionDict, IntPNames, Float64Names, BoolNames, Shape, ScalarOrArray

T1 = TypeVar("T1")
T2 = TypeVar("T2")
ST = TypeVar("ST", bound=FromResult[Any, Any])
FT = TypeVar("FT", bound=Callable[..., Any])
RT = TypeVar("RT", bound=Result)

class FromResult(Generic[FT, RT], types.FunctionType, metaclass=ABCMeta):
    REDUCTION_NAMES: ClassVar[ReductionDict]
    def __init__(
        self,
        func: FT,
        name: str,
        module: Optional[str] = None,
        doc: Optional[str] = None,
    ) -> None: ...

    __name__: str
    __qualname__: str
    __module__: str
    __doc__: Optional[str]
    __annotations__: types.MappingProxyType[str, Any]  # type: ignore[assignment]
    __signature__: Optional[inspect.Signature]
    __text_signature__: Optional[str]
    __closure__: Optional[Tuple[types._Cell, ...]]
    __defaults__: Optional[Tuple[Any, ...]]
    __globals__: types.MappingProxyType[str, Any]  # type: ignore[assignment]
    __kwdefaults__: types.MappingProxyType[str, Any]  # type: ignore[assignment]
    @property
    def __code__(self) -> types.CodeType: ...  # type: ignore[override]
    def __get__(self, obj: Optional[object], type: Optional[type]) -> types.MethodType: ...

    __call__: FT
    def __hash__(self) -> int: ...
    def __eq__(self, value: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __reduce__(self: ST) -> Tuple[Type[ST], Tuple[FT, str, str], Optional[str]]: ...
    def __setstate__(self, state: Optional[str]) -> None: ...
    @overload
    @abstractmethod
    def from_result(self: FromResult[Callable[..., T1], RT], result: RT, reduce: None = ..., axis: None = ...) -> T1: ...
    @overload
    @abstractmethod
    def from_result(self: FromResult[Callable[..., T1], RT], result: RT, reduce: Callable[[T1], T2], axis: None = ...) -> T2: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduce: Float64Names, axis: None = ...) -> np.float64: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduce: Float64Names, axis: Shape) -> ScalarOrArray[np.float64]: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduce: IntPNames, axis: None = ...) -> np.intp: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduce: IntPNames, axis: Shape) -> ScalarOrArray[np.intp]: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduce: BoolNames, axis: None = ...) -> np.bool_: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduce: BoolNames, axis: Shape) -> ScalarOrArray[np.bool_]: ...

    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: None, axis: None = ...) -> T1: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: Callable[[T1], T2], axis: None = ...) -> T2: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: Float64Names, axis: None = ...) -> np.float64: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: Float64Names, axis: Shape) -> ScalarOrArray[np.float64]: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: IntPNames, axis: None = ...) -> np.intp: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: IntPNames, axis: Shape) -> ScalarOrArray[np.intp]: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: BoolNames, axis: None = ...) -> np.bool_: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduce: BoolNames, axis: Shape) -> ScalarOrArray[np.bool_]: ...

    @staticmethod
    def _pop(dct: MutableMapping[str, T1], key: str, callback: Callable[[], T1]) -> T1: ...

@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: None = ..., axis: None = ...) -> Any: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: Callable[[Any], T1], axis: None = ...) -> T1: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: Float64Names, axis: None = ...) -> np.float64: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: Float64Names, axis: Shape) -> ScalarOrArray[np.float64]: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: IntPNames, axis: None = ...) -> np.intp: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: IntPNames, axis: Shape) -> ScalarOrArray[np.intp]: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: BoolNames, axis: None = ...) -> np.bool_: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduce: BoolNames, axis: Shape) -> ScalarOrArray[np.bool_]: ...

@overload
def call_method(obj: object, name: str, *args: Any, reduce: None = ..., axis: None = ..., **kwargs: Any) -> Any: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: Callable[[Any], T1], axis: None = ..., **kwargs: Any) -> T1: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: Float64Names, axis: None = ..., **kwargs: Any) -> np.float64: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: Float64Names, axis: Shape, **kwargs: Any) -> ScalarOrArray[np.float64]: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: IntPNames, axis: None = ..., **kwargs: Any) -> np.intp: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: IntPNames, axis: Shape, **kwargs: Any) -> ScalarOrArray[np.intp]: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: BoolNames, axis: None = ..., **kwargs: Any) -> np.bool_: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduce: BoolNames, axis: Shape, **kwargs: Any) -> ScalarOrArray[np.bool_]: ...
