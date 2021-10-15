"""Functions for calculating the pressure."""

from __future__ import annotations

from typing import Callable, Any, TYPE_CHECKING

import numpy as np
from scipy import constants
from scm.plams import Units
from qmflows.packages.cp2k_package import CP2K_Result
from qmflows.warnings_qmflows import QMFlows_Warning
from nanoutils import warning_filter

from . import FromResult
from ..utils import slice_iter

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike
    from numpy import float64 as f8

__all__ = ['get_pressure']


@FromResult
def get_pressure(
    forces: ArrayLike,
    coords: ArrayLike,
    volume: ArrayLike,
    temp: float = 298.15,
    *,
    forces_unit: str = 'ha/bohr',
    coords_unit: str = 'bohr',
    volume_unit: str = 'bohr',
    return_unit: str = 'ha/bohr^3',
) -> NDArray[f8]:
    r"""Calculate the pressure from the passed **forces**.

    .. math::

        P = \frac{Nk_{B}T}{V} + \frac{1}{6V}
            \sum_i^N \sum_j^N {\boldsymbol{r}_{ij} \cdot \boldsymbol{f}_{ij}}

    Parameters
    ----------
    forces : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{\text{mol}}, n_{\text{atom}}, 3)`
        A 3D array containing the forces of all molecules within the trajectory.
    coords : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{\text{mol}}, n_{\text{atom}}, 3)`
        A 3D array containing the coordinates of all molecules within the trajectory.
    volume : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{\text{mol}},)`
        A 1D array containing the cell volumes across the trajectory.
    temp : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{\text{mol}},)`
        A 1D array of the temperatures across the trajectory.
    forces_unit : :class:`str`
        The unit of the **forces**.
    coords_unit : :class:`str`
        The unit of the **coords**.
    volume_unit : :class:`str`
        The unit of the **volume**.
        The passed unit will automatically cubed, *e.g.* ``Angstrom -> Angstrom**3``.
    return_unit : :class:`str`
        The unit of the to-be returned pressure.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(n_{\text{mol}},)`
        A 1D array with all pressures across the trajectory.


    .. automethod:: get_pressure.from_result

    """  # noqa: E501
    _f = np.asarray(forces, dtype=np.float64) * Units.conversion_ratio(forces_unit, 'ha/bohr')
    xyz = np.asarray(coords, dtype=np.float64) * Units.conversion_ratio(coords_unit, 'bohr')
    v = np.asarray(volume, dtype=np.float64) * Units.conversion_ratio(volume_unit, 'bohr')**3
    t = np.asarray(temp, dtype=np.float64)
    k = Units.convert(constants.Boltzmann, 'j', 'hartree')

    a = (xyz.shape[-2] * k * t) / v
    b = np.empty(len(xyz), dtype=np.float64)

    shape = xyz.shape + (_f.shape[1],)
    iterator = slice_iter(shape, itemsize=b.dtype.itemsize)
    for slc in iterator:
        f = _f[slc, ..., None, :] + _f[slc, ..., None, :, :]
        r = abs(xyz[slc, ..., None, :] - xyz[slc, ..., None, :, :])
        b[slc] = np.einsum("...ijk,...ijk->...ij", r, f).sum(axis=(-1, -2))

    b /= 6 * v
    return (a + b) * Units.conversion_ratio('ha/bohr^3', return_unit)


@get_pressure._set_result_func
@warning_filter('error', category=QMFlows_Warning)
def _(
    self: FromResult[Callable[..., Any]],
    result: CP2K_Result,
    *,
    reduce: None | str | Callable[[Any], Any] = None,
    axis: None | int | tuple[int, ...] = None,
    return_unit: str = 'ha/bohr^3',
    **kwargs: Any,
) -> Any:
    r"""Call :func:`get_pressure` using argument extracted from **result**.

    Parameters
    ----------
    result : :class:`qmflows.CP2K_Result <qmflows.packages.cp2k_package.CP2K_Result>`
        The Result instance that **self** should operator on.
    reduce : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
        A callback for reducing the output of **self**.
        Alternativelly, one can provide on of the string aliases
        from :attr:`~FromResult.REDUCTION_NAMES`.
    axis : :class:`int` or :class:`Sequence[int] <collections.abc.Sequence>`, optional
        The axis along which the reduction should take place.
        If :data:`None`, use all axes.
    return_unit : :class:`str`
        The unit of the to-be returned quantity.
    \**kwargs : :data:`~typing.Any`
        Further keyword arguments for :func:`get_pressure`.

    Returns
    -------
    :data:`~typing.Any`
        The output of :func:`get_pressure`.

    """  # noqa: E501
    if result.status in {'failed', 'crashed'}:
        raise RuntimeError(f"Cannot extract data from a job with status {result.status!r}")

    a_to_au = Units.conversion_ratio('angstrom', 'bohr')
    forces = self._pop(
        kwargs, 'forces',
        callback=lambda: getattr(result, 'forces'),
    )
    temp = self._pop(
        kwargs, 'temp',
        callback=lambda: getattr(result, 'temperature'),
    )
    coords = self._pop(
        kwargs, 'coords',
        callback=lambda: getattr(result, 'coordinates') * a_to_au,
    )
    volume = self._pop(
        kwargs, 'volume',
        callback=lambda: getattr(result, 'volume') * a_to_au**3,
    )

    ret = self(forces, coords, volume, temp, return_unit=return_unit, **kwargs)
    return self._reduce(ret, reduce, axis)
