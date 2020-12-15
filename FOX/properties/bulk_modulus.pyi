import sys
from typing import TypeVar, Callable, Any, overload, Union

import numpy as np
import numpy.typing as npt
from qmflows.packages.cp2k_package import CP2K_Result

from FOX.properties import FromResult

if sys.version_info >= (3, 8):
    from typing import TypedDict, Literal
else:
    from typing_extensions import TypedDict, Literal

T1 = TypeVar("T1")
T2 = TypeVar("T2")
FT = TypeVar("FT", bound=Callable[..., Any])

def get_bulk_modulus(
    pressure: npt.ArrayLike,
    volume: npt.ArrayLike,
    pressure_unit: str = ...,
    volume_unit: str = ...,
    return_unit: str = ...,
) -> Union[np.float64, np.ndarray[Any, np.dtype[np.float64]]]: ...

class GetBulkMod(FromResult[FT, CP2K_Result]):
    @overload
    def from_result(self: GetBulkMod[Callable[..., T1]], result: CP2K_Result, reduction: None = ..., *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> T1: ...
    @overload
    def from_result(self: GetBulkMod[Callable[..., T1]], result: CP2K_Result, reduction: Callable[[T1], T2], *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> T2: ...
    @overload
    def from_result(self, result: CP2K_Result, reduction: Literal['min', 'max', 'mean', 'sum', 'product', 'var', 'std', 'ptp'], *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> np.float64: ...
    @overload
    def from_result(self, result: CP2K_Result, reduction: Literal['all', 'any'], *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> np.bool_: ...
