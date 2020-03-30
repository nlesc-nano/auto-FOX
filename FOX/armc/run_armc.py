import os
from typing import Type, Union, ContextManager, Tuple, TYPE_CHECKING, Optional, Iterable, Mapping
from pathlib import Path
from contextlib import redirect_stdout

from scm.plams import init, finish, config

from .guess import guess_param
from ..logger import Plams2Logger, wrap_plams_logger

if TYPE_CHECKING:
    from .armc import ARMC
    from ..io import PSFContainer
else:
    from ..type_alias import ARMC, PSFContainer


class Init(ContextManager[None]):
    """A context manager for calling :func:`init` and :func:`finish`."""

    def __init__(self, path: Union[None, str, os.PathLike] = None,
                 folder: Union[None, str, os.PathLike] = None) -> None:
        self.path = path
        self.folder = folder

    def __enter__(self) -> None:
        init(self.path, self.folder)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        finish()


def from_yaml(obj_type: Type[ARMC], filename: str) -> Tuple[ARMC, dict]:
    """Create a :class:`.ARMC` instance from a .yaml file.

    Parameters
    ----------
    filename : str
        The path+filename of a .yaml file containing all :class:`ARMC` settings.

    Returns
    -------
    |FOX.ARMC|_ and |dict|_
        A new :class:`ARMC` instance and
        a dictionary with keyword arguments for :func:`.run_armc`.

    """
    return NotImplemented


def run_armc(armc: ARMC,
             path: Union[None, str, os.PathLike] = None,
             folder: Union[None, str, os.PathLike] = None,
             logfile: Union[None, str, os.PathLike] = None,
             psf: Optional[Iterable[PSFContainer]] = None,
             restart: bool = False,
             guess: Optional[Mapping[str, Mapping]] = None) -> None:
    """A wrapper arround :class:`ARMC` for handling the JobManager."""
    # Create a .psf file if specified
    if psf is not None:
        for item in psf:
            item.write(None)

    # Guess the remaining unspecified parameters based on either UFF or the RDF
    if guess is not None:
        for k, v in guess.items():
            frozen = k if v['frozen'] else None
            guess_param(armc, mode=v['mode'], columns=k, frozen=frozen)

    # Initialize the ARMC procedure
    with Init(path=path, folder=folder):
        # Create the .psf file
        if psf is not None:
            workdir = Path(config.default_jobmanager.workdir)
            for i, psf_obj in enumerate(psf):
                psf_obj.write(workdir / f'mol.{i}.psf')

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
