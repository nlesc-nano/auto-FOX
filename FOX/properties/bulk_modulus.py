"""Functions for calculating the bulk modulus."""

from pathlib import Path
from typing import Union, overload, TypeVar, Callable, Any, TYPE_CHECKING

import numpy as np
from scm.plams import Units
from nanoutils import ArrayLike, Literal
from qmflows.packages.cp2k_package import CP2K_Result

from . import FromResult, get_pressure
from ..io import read_volumes

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

        B = -V * \frac{\delta P}{\delta V}

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

    if not TYPE_CHECKING:
        @property
        def __call__(self):  # noqa: D205,D400
            """
            Note
            ----
            Using :meth:`get_bulk_modulus.from_result <FromResult.from_result>` requires the passed
            :class:`qmflows.CP2K_Result <qmflows.packages.cp2k_package.CP2K_Result>`
            to have access to the following files for each argument:

            * **pressure**: ``cp2k-frc-1.xyz``, ``cp2k-pos-1.xyz``, ``cp2k-1.cell`` & ``cp2k-1.ener``
            * **volume**: ``cp2k-1.cell``

            Furthermore, in order to get sensible results both the pressure and
            cell volume must be variable.

            """  # noqa: E501
            return self._func

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
            raise RuntimeError(f"Cannot extract data a job with status {result.status!r}")
        else:
            base = Path(result.archive['work_dir'])  # type: ignore[arg-type]

        volume = self._pop(
            kwargs, 'volume',
            callback=lambda: read_volumes(base / 'cp2k-1.cell', unit='bohr')
        )
        pressure = get_pressure.from_result(result, reduction=None, volume=volume)

        ret = self(pressure, volume, **kwargs)
        return self._reduce(ret, reduction)
