"""Functions for calculating the bulk modulus."""

from pathlib import Path
from typing import Union, overload, TypeVar, Callable, Any

import numpy as np
from scm.plams import Units
from nanoutils import ArrayLike, Literal
from qmflows.packages.cp2k_package import CP2K_Result

from . import FromResult, get_pressure
from ..io import read_multi_xyz, read_volumes, read_temperatures

__all__ = ['get_bulk_modulus', 'GetBulkMod']

T = TypeVar("T")
T1 = TypeVar("T1")
FT = TypeVar("FT", bound=Callable[..., Any])


def get_bulk_modulus(
    pressure: ArrayLike,
    volume: ArrayLike,
    pressure_unit: str = 'ha/bohr^3',
    volume_unit: str = 'bohr',
    return_unit: str = 'ha/bohr^3',
) -> Union[np.float64, np.ndarray]:
    r"""Calculate the bulk modulus via differentiation of **pressure** w.r.t. **volume**.

    .. math::

        B = -V_{0} * \frac{\delta P}{\delta V}

    Parameters
    ----------
    pressure : :class:`np.ndarray[np.float64] <numpy.ndarray>`
        A 1D array of pressures used for defining :math:`\delta P`.
        Must be of equal length as **volume**.
    volume : :class:`np.ndarray[np.float64] <numpy.ndarray>`
        A 1D array of volumes used for defining :math:`\delta V`.
        Must be of equal length as **pressure**.
    pressure_unit : :class:`str`
        The unit of the **pressure**.
    volume_unit : :class:`str`
        The unit of the **volume**.
        The passed unit will automatically cubed, *e.g.* ``Angstrom -> Angstrom**3``.
    return_unit : :class:`str`
        The unit of the to-be returned pressure.

    Returns
    -------
    :class:`np.float64 <numpy.float64>` or :class:`np.ndarray[np.float64] <numpy.ndarray>`
        The bulk modulus :math:`B`.
        Returend as either a scalar or array, depending on the dimensionality **volume_ref**.

    """
    # Parse `pressure` and `volume`
    p = np.asarray(pressure, dtype=np.float64) * Units.conversion_ratio(pressure_unit, 'ha/bohr^3')
    v = np.asarray(volume, dtype=np.float64) * Units.conversion_ratio(volume_unit, 'bohr')**3

    ret = np.gradient(p, v)
    ret *= -v
    ret *= Units.conversion_ratio('ha/bohr^3', return_unit)
    return ret


class GetBulkMod(FromResult[FT, CP2K_Result]):
    """A :class:`FOX.properties.FromResult` subclass for :func:`get_bulk_modulus`."""

    @overload
    def from_result(self: FromResult[Callable[..., T], CP2K_Result], result: CP2K_Result, reduction: None = ..., **kwargs: Any) -> T: ...  # noqa: E501
    @overload
    def from_result(self: FromResult[Callable[..., T], CP2K_Result], result: CP2K_Result, reduction: Callable[[T], T1], **kwargs: Any) -> T1: ...  # noqa: E501
    @overload
    def from_result(self, result: CP2K_Result, reduction: Literal['min', 'max', 'mean', 'sum', 'product', 'var', 'std', 'ptp'], **kwargs: Any) -> np.float64: ...  # noqa: E501
    @overload
    def from_result(self, result: CP2K_Result, reduction: Literal['all', 'any'], **kwargs: Any) -> np.bool_: ...  # noqa: E501
    def from_result(self, result, reduction=None, **kwargs: Any):   # noqa: E301
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

        pressure = get_pressure(
            forces, coords, volume, temp, coords_unit='angstrom', volume_unit='angstrom'
        )
        ret = self(pressure, volume, **kwargs)
        return self._reduce(ret, reduction)
