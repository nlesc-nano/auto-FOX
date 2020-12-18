from typing import TypeVar, Callable, Any, overload, Union

import numpy as np
import numpy.typing as npt
from qmflows.packages.cp2k_package import CP2K_Result

from FOX.properties import FromResult
from FOX.properties.annotations import IntPNames, Float64Names, BoolNames, Shape, ScalarOrArray

T1 = TypeVar("T1")
T2 = TypeVar("T2")
FT = TypeVar("FT", bound=Callable[..., Any])

def get_bulk_modulus(
    pressure: npt.ArrayLike,
    volume: npt.ArrayLike,
    pressure_unit: str = ...,
    volume_unit: str = ...,
    return_unit: str = ...,
) -> ScalarOrArray[np.float64]: ...

class GetBulkMod(FromResult[FT, CP2K_Result]):
    @overload
    def from_result(self: GetBulkMod[Callable[..., T1]], result: CP2K_Result, reduce: None = ..., shape: None = ..., *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> T1: ...
    @overload
    def from_result(self: GetBulkMod[Callable[..., T1]], result: CP2K_Result, reduce: Callable[[T1], T2], shape: None = ..., *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> T2: ...
    @overload
    def from_result(self, result: CP2K_Result, reduce: Float64Names, shape: None = ..., *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> np.float64: ...
    @overload
    def from_result(self, result: CP2K_Result, reduce: Float64Names, shape: Shape, *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> ScalarOrArray[np.float64]: ...
    @overload
    def from_result(self, result: CP2K_Result, reduce: IntPNames, shape: None = ..., *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> np.intp: ...
    @overload
    def from_result(self, result: CP2K_Result, reduce: IntPNames, shape: Shape, *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> ScalarOrArray[np.intp]: ...
    @overload
    def from_result(self, result: CP2K_Result, reduce: BoolNames, shape: None = ..., *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> np.bool_: ...
    @overload
    def from_result(self, result: CP2K_Result, reduce: BoolNames, shape: Shape, *, pressure: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., pressure_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> ScalarOrArray[np.bool_]: ...
