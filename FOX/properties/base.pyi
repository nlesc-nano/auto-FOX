import types
import inspect
from abc import ABCMeta, abstractmethod
from typing import Generic, Callable, Any, ClassVar, TypeVar, overload, Optional, MutableMapping, Mapping, Tuple

from qmflows.packages import Result
import numpy as np
import numpy.typing as npt

from FOX.properties.annotations import ReductionDict, IntPNames, Float64Names, BoolNames

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
    @property
    def __annotations__(self) -> Optional[Mapping[str, Any]]: ...  # type: ignore[override]
    @property
    def __signature__(self) -> Optional[inspect.Signature]: ...
    @property
    def __text_signature__(self) -> Optional[str]: ...
    @property
    def __closure__(self) -> Optional[Tuple[types._Cell, ...]]: ...  # type: ignore[override]
    @property
    def __code__(self) -> types.CodeType: ...  # type: ignore[override]
    @property
    def __defaults__(self) -> Optional[Tuple[Any, ...]]: ...  # type: ignore[override]
    @property
    def __globals__(self) -> Mapping[str, Any]: ...  # type: ignore[override]
    @property
    def __kwdefaults__(self) -> Mapping[str, Any]: ...  # type: ignore[override]

    __call__: FT
    def __get__(self, obj: Optional[object], type: Optional[type]) -> types.MethodType: ...
    def __hash__(self) -> int: ...
    def __eq__(self, value: object) -> bool: ...
    def __repr__(self) -> str: ...
    @overload
    @abstractmethod
    def from_result(self: FromResult[Callable[..., T1], RT], result: RT, reduction: None = ...) -> T1: ...
    @overload
    @abstractmethod
    def from_result(self: FromResult[Callable[..., T1], RT], result: RT, reduction: Callable[[T1], T2]) -> T2: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduction: Float64Names) -> np.float64: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduction: IntPNames) -> np.intp: ...
    @overload
    @abstractmethod
    def from_result(self, result: RT, reduction: BoolNames) -> np.bool_: ...

    @overload
    @classmethod
    def _reduce(cls, value: T1, reduction: None) -> T1: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduction: Callable[[T1], T2]) -> T2: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduction: Float64Names) -> np.float64: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduction: IntPNames) -> np.intp: ...
    @overload
    @classmethod
    def _reduce(cls, value: T1, reduction: BoolNames) -> np.bool_: ...

    @staticmethod
    def _pop(dct: MutableMapping[str, T1], key: str, callback: Callable[[], T1]) -> T1: ...

@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduction: None = ...) -> Any: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduction: Callable[[Any], T1]) -> T1: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduction: Float64Names) -> np.float64: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduction: IntPNames) -> np.intp: ...
@overload
def get_attr(obj: object, name: str, default: Any = ..., *, reduction: BoolNames) -> np.bool_: ...

@overload
def call_method(obj: object, name: str, *args: Any, reduction: None = ..., **kwargs: Any) -> Any: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduction: Callable[[Any], T1], **kwargs: Any) -> T1: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduction: Float64Names, **kwargs: Any) -> np.float64: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduction: IntPNames, **kwargs: Any) -> np.intp: ...
@overload
def call_method(obj: object, name: str, *args: Any, reduction: BoolNames, **kwargs: Any) -> np.bool_: ...
