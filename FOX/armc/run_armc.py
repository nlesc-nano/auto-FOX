import os
from typing import Type, Union, ContextManager, Tuple, TYPE_CHECKING, Optional, Iterable, Mapping
from pathlib import Path
from contextlib import redirect_stdout

from scm.plams import init, finish, config
from qmflows.utils import InitRestart

from .guess import guess_param
from ..logger import Plams2Logger, wrap_plams_logger

if TYPE_CHECKING:
    from .armc import ARMC
    from ..io import PSFContainer
else:
    from ..type_alias import ARMC, PSFContainer


def run_armc(armc: ARMC,
             path: Union[None, str, os.PathLike] = None,
             folder: Union[None, str, os.PathLike] = None,
             logfile: Union[None, str, os.PathLike] = None,
             psf: Optional[Iterable[PSFContainer]] = None,
             restart: bool = False,
             guess: Optional[Mapping[str, Mapping]] = None) -> None:
    """A wrapper arround :class:`ARMC` for handling the JobManager."""
    # Guess the remaining unspecified parameters based on either UFF or the RDF
    if guess is not None:
        for k, v in guess.items():
            frozen = k if v['frozen'] else None
            guess_param(armc, mode=v['mode'], columns=k, frozen=frozen)

    # Initialize the ARMC procedure
    with InitRestart(path=path, folder=folder):
        # Create the .psf file
        if psf is not None:
            workdir = Path(config.default_jobmanager.workdir)
            for i, psf_obj in enumerate(psf):
                psf_obj.write(workdir / f'mol.{i}.psf')

        armc()
        return None

        # Create the logger
        armc.logger = wrap_plams_logger(logfile, armc.__class__.__name__)
        writer = Plams2Logger(armc.logger,
                              lambda n: 'STARTED' in n,
                              lambda n: 'Renaming' in n,
                              lambda n: 'Trying to obtain results of crashed or failed job' in n)

        with redirect_stdout(writer):
            if not restart:  # To restart or not? That's the question
                armc()
            else:
                armc.restart()
