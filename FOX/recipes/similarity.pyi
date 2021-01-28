import sys
from typing import Any, Union, Callable, overload, TypeVar, Iterable

from FOX import MultiMolecule
import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 8):
    from typing import Protocol, Literal
else:
    from typing_extensions import Protocol, Literal

_SCT1 = TypeVar("_SCT1", bound=np.generic)
_SCT2 = TypeVar("_SCT2", bound=np.generic)
_SCT_co = TypeVar("_SCT_co", covariant=True, bound=np.generic)

_NDArray = np.ndarray[Any, np.dtype[_SCT_co]]
_ReduceCallBack = Callable[[_NDArray[_SCT1]], Union[_SCT2, _NDArray[_SCT2]]]

class _MetricCallBack1(Protocol[_SCT_co]):
    def __call__(
        self, __md: MultiMolecule, __md_ref: MultiMolecule,
    ) -> _NDArray[_SCT_co]: ...

class _MetricCallBack2(Protocol[_SCT_co]):
    def __call__(
        self, __md: MultiMolecule, __md_ref: MultiMolecule, **kwargs: Any,
    ) -> _NDArray[_SCT_co]: ...

_MetricCallBack = Union[_MetricCallBack1[_SCT_co], _MetricCallBack2[_SCT_co]]

_MetricAliases = Literal[
    'braycurtis',
    'canberra',
    'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch',
    'cityblock', 'cblock', 'cb', 'c',
    'correlation', 'co',
    'cosine', 'cos',
    'dice',
    'euclidean', 'euclid', 'eu', 'e',
    'matching', 'hamming', 'hamm', 'ha', 'h',
    'jaccard', 'jacc', 'ja', 'j',
    'jensenshannon', 'js',
    'kulsinski',
    'mahalanobis', 'mahal', 'mah',
    'minkowski', 'mi', 'm', 'pnorm',
    'rogerstanimoto',
    'russellrao',
    'seuclidean', 'se', 's',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean', 'sqe', 'sqeuclid',
    'yule',
]

@overload
def compare_trajectories(
    md: npt.ArrayLike,
    md_ref: npt.ArrayLike,
    *,
    metric: _MetricAliases = ...,
    reduce: None | _ReduceCallBack[np.float64, np.float64] = ...,
    reset_origin: bool = True,
    **kwargs: Any,
) -> _NDArray[np.float64]: ...
@overload
def compare_trajectories(
    md: npt.ArrayLike,
    md_ref: npt.ArrayLike,
    *,
    metric: _MetricCallBack[_SCT1],
    reduce: None | _ReduceCallBack[_SCT1, _SCT1] = ...,
    reset_origin: bool = True,
    **kwargs: Any,
) -> _NDArray[_SCT1]: ...
@overload
def compare_trajectories(
    md: npt.ArrayLike,
    md_ref: npt.ArrayLike,
    *,
    metric: _MetricCallBack[_SCT1],
    reduce: _ReduceCallBack[_SCT1, _SCT2],
    reset_origin: bool = True,
    **kwargs: Any,
) -> _NDArray[_SCT2]: ...
@overload
def compare_trajectories(
    md: npt.ArrayLike,
    md_ref: npt.ArrayLike,
    *,
    reduce: _ReduceCallBack[np.float64, _SCT2],
    metric: _MetricAliases = ...,
    reset_origin: bool = True,
    **kwargs: Any,
) -> _NDArray[_SCT2]: ...

def fps_reduce(
    dist_mat: _NDArray[np.number[Any] | np.bool_ | np.character | np.object_],
    n: None | int = ...,
    *,
    operation: Literal["min", "max"] = ...,
    cluster_size: int | Iterable[int] = ...,
    start: None | int = ...,
    randomness: None | float = ...,
    weight: Callable[[np.ndarray], np.ndarray] = ...,
) -> _NDArray[np.intp]: ...
