"""Functions for calculating the pressure."""

from pathlib import Path
from typing import TypeVar, Callable, Any, overload

import numpy as np
from scipy import constants
from scm.plams import Units
from qmflows.packages.cp2k_package import CP2K_Result
from nanoutils import ArrayLike, Literal

from . import FromResult
from ..io import read_multi_xyz, read_temperatures, read_volumes

__all__ = ['get_pressure', 'GetPressure']

T = TypeVar("T")
T1 = TypeVar("T1")
FT = TypeVar("FT", bound=Callable[..., Any])


def get_pressure(
    forces: ArrayLike,
    coords: ArrayLike,
    volume: ArrayLike,
    temp: ArrayLike = 298.15,
    forces_unit: str = 'ha/bohr',
    coords_unit: str = 'bohr',
    volume_unit: str = 'bohr',
    return_unit: str = 'ha/bohr^3',
) -> np.ndarray:
    r"""Calculate the pressure from the passed **forces**.

    .. math::

        P = \frac{NK_{B}T}{V} + \frac{1}{6V} \sum_{i}\sum_{{j}\ne{i}}{r_{ij} * f_{ij}}

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

    """  # noqa: E501
    _f = np.asarray(forces, dtype=np.float64) * Units.conversion_ratio(forces_unit, 'ha/bohr')
    xyz = np.asarray(coords, dtype=np.float64) * Units.conversion_ratio(coords_unit, 'bohr')
    v = np.asarray(volume, dtype=np.float64) * Units.conversion_ratio(volume_unit, 'bohr')**3
    t = np.asarray(temp, dtype=np.float64)
    k = Units.convert(constants.Boltzmann, 'joule', 'hartree')

    f = _f[..., None, :] + _f[..., None, :, :]
    r = xyz[..., None, :] - xyz[..., None, :, :]

    a = (xyz.shape[-2] * k * t) / v
    b = (1 / (6 * v)) * np.linalg.norm((r * f).sum(axis=(-2, -3)), axis=-1)
    return (a + b) * Units.conversion_ratio('ha/bohr^3', return_unit)


class GetPressure(FromResult[FT, CP2K_Result]):
    """A :class:`FOX.properties.FromResult` subclass for :func:`get_pressure`."""

    @overload
    def from_result(self: FromResult[Callable[..., T], CP2K_Result], result: CP2K_Result, reduction: None = ..., **kwargs: Any) -> T: ...  # noqa: E501
    @overload
    def from_result(self: FromResult[Callable[..., T], CP2K_Result], result: CP2K_Result, reduction: Callable[[T], T1], **kwargs: Any) -> T1: ...  # noqa: E501
    @overload
    def from_result(self, result: CP2K_Result, reduction: Literal['min', 'max', 'mean', 'sum', 'product', 'var', 'std', 'ptp'], **kwargs: Any) -> np.float64: ...  # noqa: E501
    @overload
    def from_result(self, result: CP2K_Result, reduction: Literal['all', 'any'], **kwargs: Any) -> np.bool_: ...  # noqa: E501
    def from_result(self, result, reduction=None, **kwargs: Any):  # noqa: E301
        r"""Call **self** using argument extracted from **result**.

        Parameters
        ----------
        result : :class:`qmflows.CP2K_Result <qmflows.packages.cp2k_package.CP2K_Result>`
            The Result instance that **self** should operator on.
        reduction : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
            A callback for reducing the output of **self**.
            Alternativelly, one can provide on of the string aliases from :attr:`REDUCTION_NAMES`.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for :meth:`__call__`.

        Returns
        -------
        :data:`~typing.Any`
            The output of :meth:`__call__`.

        """  # noqa: E501
        if result.status in {'failed', 'crashed'}:
            raise RuntimeError("Cannot extract data a job with status {result.status!r}")
        else:
            base = Path(result.archive['workdir'])  # type: ignore[arg-type]

        forces, _ = read_multi_xyz(base / 'cp2k-frc-1.xyz', return_comment=False)
        coords, _ = read_multi_xyz(base / 'cp2k-pos-1.xyz', return_comment=False)
        volume = read_volumes(base / 'cp2k-1.cell')
        temp = read_temperatures(base / 'cp2k-1.PARTICLES.temp')

        ret = self(
            forces, coords, volume, temp, coords_unit='angstrom', volume_unit='angstrom', **kwargs
        )
        return self._reduce(ret, reduction)
