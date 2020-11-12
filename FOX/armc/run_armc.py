"""A module holsing the :func:`run_armc` function.

Index
-----
.. currentmodule:: FOX.armc
.. autosummary::
    run_armc

API
---
.. autofunction:: run_armc

"""

import os
from typing import Union, TYPE_CHECKING, Optional, Iterable
from pathlib import Path
from contextlib import redirect_stdout

from scm.plams import config
from qmflows.utils import InitRestart

from ..utils import log_traceback_locals
from ..logger import Plams2Logger, wrap_plams_logger

if TYPE_CHECKING:
    from . import MonteCarloABC
    from ..io import PSFContainer
else:
    from ..type_alias import MonteCarloABC, PSFContainer

__all__ = ['run_armc']


def run_armc(armc: MonteCarloABC,
             path: Union[None, str, os.PathLike] = None,
             folder: Union[None, str, os.PathLike] = None,
             logfile: Union[None, str, os.PathLike] = None,
             psf: Optional[Iterable[PSFContainer]] = None,
             restart: bool = False) -> None:
    """A wrapper arround :class:`MonteCarloABC` for handling the JobManager."""
    # Initialize the ARMC procedure
    with InitRestart(path=path, folder=folder):
        # Create the .psf file
        if psf is not None:
            workdir = Path(config.default_jobmanager.workdir)
            for i, psf_obj in enumerate(psf):
                psf_obj.write(workdir / f'mol.{i}.psf')

        # Disable rerun prevention
        config.default_jobmanager.settings.hashing = None

        # Create the logger
        armc.logger = wrap_plams_logger(logfile, f'{armc.__class__.__name__}_{id(armc)}')
        writer = Plams2Logger(armc.logger,
                              lambda n: 'STARTED' in n,
                              lambda n: 'Renaming' in n,
                              lambda n: 'Trying to obtain results of crashed or failed job' in n)

        with redirect_stdout(writer):
            try:
                if not restart:  # To restart or not? That's the question
                    armc()
                else:
                    armc.restart()
            except Exception:
                logger = armc.logger
                logger.debug("Unexpected exception encounterd, dumping local variables:")
                log_traceback_locals(logger)
                raise
