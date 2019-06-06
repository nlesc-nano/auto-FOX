"""A module with functions for analyzing ARMC results."""

from typing import Optional

import pandas as pd

from FOX.io.hdf5_utils import from_hdf5
from FOX.functions.utils import assert_error

try:
    import matplotlib.pyplot as plt
    __all__ = ['compare_pes_descriptors']
    PltFigure = plt.Figure
    PLT_ERROR = ''
except ImportError:
    __all__ = []
    PltFigure = 'matplotlib.pyplot.Figure'
    PLT_ERROR = ("Use of the FOX.{} function requires the 'matplotlib' package."
                 "\n'matplotlib' can be installed via PyPi with the following command:"
                 "\n\tpip install matplotlib")


@assert_error(PLT_ERROR)
def compare_pes_descriptors(filename_in: str,
                            filename_out: str,
                            descriptor: str,
                            iteration: int = -1,
                            savefig_kwarg: Optional[dict] = None) -> plt.Figure:
    """Create and save a figure showing side-by-side comparisons of QM and MM PES descriptors.

    .. _matplotlib.savefig: https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.savefig.html

    Examples
    --------
    .. image:: rdf.png
        :align: center

    Parameters
    ----------
    filename_in : str
        The path+name of the ARMC .hdf5 file.

    filename_out : str
        The path+name of the to-be created image.

    descriptor : str
        The name of the dataset containing the to-be retrieved PES descriptor (*e.g.* ``"RDF"``).

    iteration : int
        The ARMC iteration containg the PES descriptor of interest.

    savefig_kwarg : |None|_ or |dict|_
        Optional: A dictionary with user-specified keyword arguments for matplotlib.savefig_.

    Returns
    -------
    |matplotlib.figure.Figure|_:
        A matplotlib figure containg PES descriptors.

    """
    # Gather the pes descriptors
    pes_mm = from_hdf5(filename_in, descriptor)[iteration]
    pes_qm = from_hdf5(filename_in, descriptor + '.ref')[0]

    # Define constants
    ncols = len(pes_mm.columns)
    figsize = (4 * ncols, 6)

    # Construct the figures
    fig, ax_tup = plt.subplots(ncols=ncols, sharex=True, sharey=False)
    for key, ax in zip(pes_mm, ax_tup):
        df = pd.DataFrame({'MM-MD': pes_mm[key], 'QM-MD': pes_qm[key]}, index=pes_mm.index)
        df.index.name = descriptor
        df.plot(ax=ax, title=key, figsize=figsize)

    # Save and return the figures
    if savefig_kwarg is None:
        plt.savefig(filename_out, dpi=300, quality=100, transparent=True, format='png')
    else:
        plt.savefig(filename_out, **savefig_kwarg)
    return fig
