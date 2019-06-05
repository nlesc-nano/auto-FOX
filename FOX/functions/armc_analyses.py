"""A module with functions for analyzing ARMC results."""

from typing import Optional

import h5py
import pandas as pd
import matplotlib.pyplot as plt

import FOX

filename_in = 1
filename_out = 1
descriptor = 'RDF'


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
    pes_mm = FOX.from_hdf5(filename_in, descriptor)[iteration]
    pes_qm = pes_mm.copy()
    with h5py.File(filename_in, 'r') as f:
        pes_qm[:] = f[descriptor].attrs['ref'][:]

    # Define constants
    ncols = len(pes_mm.columns)
    figsize = (ncols*4, 6)

    # Construct the figures
    fig, ax_tup = plt.subplots(ncols=ncols, sharex=True, sharey=False)
    for key, ax in zip(pes_mm, ax_tup):
        df = pd.DataFrame(
            [pes_mm[key], pes_qm[key]], index=pes_mm.index, columns=['QM-MD', 'MM-MD']
        )
        df.index.name = descriptor
        df.plot(ax=ax, title=key, figsize=figsize)

    # Save and return the figures
    if savefig_kwarg is None:
        plt.savefig(filename_out, dpi=300, quality=100, transparent=True, format='png')
    else:
        plt.savefig(filename_out, **savefig_kwarg)
    return fig
