from typing import TypeVar, Callable, Any, overload, Union

import numpy as np
import numpy.typing as npt
from qmflows.packages.cp2k_package import CP2K_Result

from FOX.properties import FromResult
from FOX.properties.annotations import IntPNames, Float64Names, BoolNames

T1 = TypeVar("T1")
T2 = TypeVar("T2")
FT = TypeVar("FT", bound=Callable[..., Any])

def get_pressure(
    forces: npt.ArrayLike,
    coords: npt.ArrayLike,
    volume: npt.ArrayLike,
    temp: npt.ArrayLike = ...,
    forces_unit: str = ...,
    coords_unit: str = ...,
    volume_unit: str = ...,
    return_unit: str = ...,
) -> Union[np.float64, np.ndarray[Any, np.dtype[np.float64]]]: ...

class GetPressure(FromResult[FT, CP2K_Result]):
    @overload
    def from_result(self: GetPressure[Callable[..., T1]], result: CP2K_Result, reduction: None = ..., *, forces: npt.ArrayLike = ..., coords: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., temp: npt.ArrayLike = ..., forces_unit: str = ..., coords_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> T1: ...
    @overload
    def from_result(self: GetPressure[Callable[..., T1]], result: CP2K_Result, reduction: Callable[[T1], T2], *, forces: npt.ArrayLike = ..., coords: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., temp: npt.ArrayLike = ..., forces_unit: str = ..., coords_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> T2: ...
    @overload
    def from_result(self, result: CP2K_Result, reduction: Float64Names, *, forces: npt.ArrayLike = ..., coords: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., temp: npt.ArrayLike = ..., forces_unit: str = ..., coords_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> np.float64: ...
    @overload
    def from_result(self, result: CP2K_Result, reduction: IntPNames, *, forces: npt.ArrayLike = ..., coords: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., temp: npt.ArrayLike = ..., forces_unit: str = ..., coords_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> np.intp: ...
    @overload
    def from_result(self, result: CP2K_Result, reduction: BoolNames, *, forces: npt.ArrayLike = ..., coords: npt.ArrayLike = ..., volume: npt.ArrayLike = ..., temp: npt.ArrayLike = ..., forces_unit: str = ..., coords_unit: str = ..., volume_unit: str = ..., return_unit: str = ...) -> np.bool_: ...
