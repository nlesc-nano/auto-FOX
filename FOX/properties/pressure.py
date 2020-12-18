"""Functions for calculating the pressure."""

from pathlib import Path
from typing import TypeVar, Callable, Any

import numpy as np
from scipy import constants
from scm.plams import Units
from qmflows.packages.cp2k_package import CP2K_Result

from . import FromResult

__all__ = ['get_pressure', 'GetPressure']

T = TypeVar("T")
T1 = TypeVar("T1")
FT = TypeVar("FT", bound=Callable[..., Any])


def get_pressure(
    forces,
    coords,
    volume,
    temp=298.15,
    forces_unit='ha/bohr',
    coords_unit='bohr',
    volume_unit='bohr',
    return_unit='ha/bohr^3',
):
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
    k = Units.convert(constants.Boltzmann, 'j', 'hartree')

    f = _f[..., None, :] + _f[..., None, :, :]
    r = xyz[..., None, :] - xyz[..., None, :, :]

    a = (xyz.shape[-2] * k * t) / v
    b = (1 / (6 * v)) * np.linalg.norm((r * f).sum(axis=(-2, -3)), axis=-1)
    return (a + b) * Units.conversion_ratio('ha/bohr^3', return_unit)


class GetPressure(FromResult[FT, CP2K_Result]):
    """A :class:`FOX.properties.FromResult` subclass for :func:`get_pressure`."""

    @property
    def __call__(self):  # noqa: D205, D400
        """
        Note
        ----
        Using :meth:`get_pressure.from_result() <FromResult.from_result>` requires the passed
        :class:`qmflows.CP2K_Result <qmflows.packages.cp2k_package.CP2K_Result>`
        to have access to the following files for each argument:

        * **forces**: ``cp2k-frc-1.xyz``
        * **coords**: ``cp2k-pos-1.xyz``
        * **volume**: ``cp2k-1.cell``
        * **temp**: ``cp2k-1.ener``

        """
        return self._func

    def from_result(self, result, reduce=None, axis=None, **kwargs):
        r"""Call **self** using argument extracted from **result**.

        Parameters
        ----------
        result : :class:`qmflows.CP2K_Result <qmflows.packages.cp2k_package.CP2K_Result>`
            The Result instance that **self** should operator on.
        reduce : :class:`str` or :class:`Callable[[Any], Any] <collections.abc.Callable>`, optional
            A callback for reducing the output of **self**.
            Alternativelly, one can provide on of the string aliases from :attr:`REDUCTION_NAMES`.
        axis : :class:`int` or :class:`Sequence[int] <collections.abc.Sequence>`, optional
            The axis along which the reduction should take place.
            If :data:`None`, use all axes.
        \**kwargs : :data:`~typing.Any`
            Further keyword arguments for :meth:`__call__`.

        Returns
        -------
        :data:`~typing.Any`
            The output of :meth:`__call__`.

        """  # noqa: E501
        if result.status in {'failed', 'crashed'}:
            raise RuntimeError(f"Cannot extract data from a job with status {result.status!r}")
        a_to_au = Units.conversion_ratio('angstrom', 'bohr')

        forces = self._pop(kwargs, 'forces', callback=lambda: getattr(result, 'forces'))
        temp = self._pop(kwargs, 'temp', callback=lambda: getattr(result, 'temperature'))
        coords = self._pop(
            kwargs, 'coords', callback=lambda: getattr(result, 'coords') * a_to_au
        )
        volume = self._pop(
            kwargs, 'volume', callback=lambda: getattr(result, 'volume') * a_to_au**3
        )

        ret = self(forces, coords, volume, temp, **kwargs)
        return self._reduce(ret, reduce, axis)
